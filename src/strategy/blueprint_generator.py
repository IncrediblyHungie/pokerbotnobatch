"""
Blueprint strategy generator using Linear MCCFR.
Runs self-play training to create a baseline strategy.
"""

import yaml
import random
import os
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Tuple
from tqdm import tqdm
from engine.game_state import GameState, BettingRound
from engine.hand_evaluator import Card, Rank, Suit, create_card
from abstraction.simple_card_abstraction import SimpleCardAbstraction
from abstraction.action_abstraction import ActionAbstraction
from cfr.linear_cfr import LinearCFR
from cfr.batch_mccfr import BatchMCCFR
from cfr.simple_cfr import SimpleCFR
from utils.device_config import setup_device


def _worker_train_batch(args: Tuple[int, int, str]):
    """Worker function to train a batch of iterations in parallel."""
    worker_id, batch_size, config_path = args

    # Import here to avoid recursive import issues when spawning processes
    from strategy.blueprint_generator import BlueprintGenerator

    # Each worker uses CPU to avoid GPU contention
    blueprint_gen = BlueprintGenerator(
        config_path,
        use_gpu=False,
        use_batch_processing=False
    )

    total_utility = {0: 0.0, 1: 0.0}

    for _ in range(batch_size):
        game_state = blueprint_gen.setup_game(2)
        utilities = blueprint_gen.cfr_solver.train_iteration(game_state)
        for pid, util in utilities.items():
            total_utility[pid] += util

    infosets_data = {}
    for key, infoset in blueprint_gen.cfr_solver.infosets.items():
        infosets_data[key] = {
            'regret_sum': infoset.regret_sum.copy(),
            'strategy_sum': infoset.strategy_sum.copy(),
            'reach_count': infoset.reach_count,
            'num_actions': infoset.num_actions
        }

    return worker_id, batch_size, infosets_data, total_utility


class BlueprintGenerator:
    def __init__(self, config_path: str, use_gpu: bool = True, device_id: int = 0,
                 use_batch_processing: bool = True, batch_size: int = None):
        self.config_path = config_path
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Setup device configuration
        self.device_config = setup_device(force_cpu=not use_gpu, device_id=device_id)
        
        # Determine batch size based on GPU memory and CPU cores
        if batch_size is None:
            if self.device_config.use_gpu:
                _, total_mem = self.device_config.get_memory_info()
                import multiprocessing
                cpu_cores = multiprocessing.cpu_count()
                
                # Scale batch size based on both GPU memory and CPU cores
                if total_mem > 80e9 and cpu_cores >= 32:  # GH200-class with many CPUs
                    batch_size = 8192  # Massive batches for supercomputing setups
                elif total_mem > 30e9 and cpu_cores >= 16:  # A100-class with many CPUs
                    batch_size = 4096
                elif total_mem > 30e9:  # High-end GPU
                    batch_size = 1024
                elif total_mem > 10e9:  # Mid-range GPU
                    batch_size = 512
                else:
                    batch_size = 256
            else:
                batch_size = 128  # CPU only
        
        # Disable batch processing by default due to stability issues
        # Users can enable it later when the BatchMCCFR implementation is fixed
        self.use_batch_processing = use_batch_processing and self.device_config.use_gpu
        self.batch_size = batch_size
        
        print(f"Using {'GPU' if self.device_config.use_gpu else 'CPU'} for computation "
              f"(backend: {self.device_config.backend})")
        
        if self.use_batch_processing:
            print(f"Batch processing enabled with batch size: {batch_size}")
        else:
            print("Batch processing disabled (using single-iteration training for stability)")
        
        # Initialize abstractions with device config
        self.card_abstraction = SimpleCardAbstraction(
            self.config['abstraction'], 
            device_config=self.device_config
        )
        self.action_abstraction = ActionAbstraction(
            self.config['abstraction']['action_fractions']
        )
        
        # Initialize CFR solver with device config
        if self.use_batch_processing:
            try:
                self.cfr_solver = BatchMCCFR(
                    self.card_abstraction, 
                    self.action_abstraction,
                    device_config=self.device_config,
                    batch_size=batch_size
                )
                print("✅ BatchMCCFR initialized successfully")
            except Exception as e:
                print(f"⚠️ BatchMCCFR initialization failed: {e}")
                print("Falling back to SimpleCFR...")
                self.use_batch_processing = False
                self.cfr_solver = SimpleCFR(
                    self.card_abstraction, 
                    self.action_abstraction,
                    device_config=self.device_config
                )
        else:
            # Check config for CFR solver type
            solver_type = self.config.get('cfr', {}).get('solver_type', 'mccfr')
            
            if solver_type == 'mccfr':
                from cfr.mccfr import MCCFR
                self.cfr_solver = MCCFR(
                    self.card_abstraction, 
                    self.action_abstraction,
                    device_config=self.device_config
                )
                print("✅ MCCFR solver initialized")
            else:
                # Fallback to SimpleCFR
                self.cfr_solver = SimpleCFR(
                    self.card_abstraction, 
                    self.action_abstraction,
                    device_config=self.device_config
                )
                print("⚠️ Using SimpleCFR fallback")
        
        # Game configuration
        self.game_config = self.config['game']
        self.training_config = self.config['training']
        
    def setup_game(self, num_players: int = 6) -> GameState:
        """Create a new game with random cards dealt"""
        game_state = GameState(
            num_players=num_players,
            starting_stack=self.game_config['starting_stack'],
            small_blind=self.game_config['small_blind'],
            big_blind=self.game_config['big_blind']
        )
        
        # Create and shuffle deck for this hand
        deck = self._create_shuffled_deck()
        game_state._deck = deck.copy()
        
        # Deal hole cards
        card_idx = 0
        for player in game_state.players:
            player.hole_cards = [deck[card_idx], deck[card_idx + 1]]
            card_idx += 2
        
        # Remove dealt hole cards from the game's deck
        game_state._deck = deck[card_idx:]
        
        # Community cards start empty (will be dealt during betting rounds)
        game_state.community_cards = []
        
        return game_state
    
    def _create_shuffled_deck(self) -> List[Card]:
        """Create and shuffle a deck of cards"""
        deck = []
        for suit in Suit:
            for rank in Rank:
                deck.append(Card(rank, suit))
        
        random.shuffle(deck)
        return deck
    
    def train_blueprint(self, iterations: int = None, 
                       checkpoint_frequency: int = None) -> Dict:
        """
        Train blueprint strategy using self-play.
        
        Args:
            iterations: Number of training iterations (default from config)
            checkpoint_frequency: How often to save checkpoints (default from config)
            
        Returns:
            Training statistics
        """
        if iterations is None:
            iterations = self.training_config['iterations']
        
        if checkpoint_frequency is None:
            checkpoint_frequency = self.training_config['checkpoint_frequency']
        
        print(f"Training blueprint for {iterations} iterations...")
        
        # Training statistics
        stats = {
            'iterations': [],
            'exploitability': [],
            'total_infosets': [],
            'avg_utility': []
        }
        
        # Training loop
        if self.use_batch_processing:
            # Try batch training, fallback to simple if it fails
            try:
                print("Attempting batch training...")
                batch_iterations = max(1, iterations // self.batch_size)
                
                import signal
                import time
                
                class TimeoutException(Exception):
                    pass
                
                def timeout_handler(signum, frame):
                    raise TimeoutException("Batch training timeout")
                
                # Set a timeout for batch training
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(30)  # 30 second timeout for first batch
                
                try:
                    for batch_idx in tqdm(range(batch_iterations), desc="Batch Training"):
                        # Run batch of iterations with timeout protection
                        try:
                            utilities = self.cfr_solver.batch_train_iteration(self.batch_size)
                            # Reset alarm for subsequent batches
                            signal.alarm(0)
                            if batch_idx == 0:  # After first successful batch, increase timeout
                                signal.alarm(60)
                        except Exception as e:
                            print(f"Batch training failed: {e}")
                            print("Falling back to simple CFR training...")
                            self.use_batch_processing = False
                            break
                        
                        # Update iteration counter
                        iteration = (batch_idx + 1) * self.batch_size
                        
                        # Collect statistics every few batches
                        if batch_idx % max(1, 1000 // self.batch_size) == 0:
                            self._collect_training_stats(iteration, utilities, stats)
                        
                        # Save checkpoint
                        if iteration % checkpoint_frequency == 0 and iteration > 0:
                            checkpoint_path = f"data/blueprints/checkpoint_{iteration}.pkl"
                            self.cfr_solver.save_strategy(checkpoint_path)
                            print(f"Saved checkpoint at iteration {iteration}")
                        
                        # Memory cleanup for large batches
                        if batch_idx % 10 == 0 and hasattr(self.cfr_solver, 'batch_memory_cleanup'):
                            self.cfr_solver.batch_memory_cleanup()
                            
                except TimeoutException:
                    print("Batch training timed out, falling back to simple CFR training...")
                    self.use_batch_processing = False
                    # Create new SimpleCFR instance for fallback
                    self.cfr_solver = SimpleCFR(
                        self.card_abstraction, 
                        self.action_abstraction,
                        device_config=self.device_config
                    )
                finally:
                    signal.alarm(0)  # Cancel any pending alarm
                        
            except Exception as e:
                print(f"Batch processing initialization failed: {e}")
                print("Falling back to simple CFR training...")
                self.use_batch_processing = False
        
        # Fallback to simple training if batch processing failed
        if not self.use_batch_processing:
            # Standard single-iteration training
            for iteration in tqdm(range(iterations), desc="Training"):
                # Use fixed game setup for speed (heads-up only)
                num_players = self.game_config.get('max_players', 2)  # Prefer max players for consistency
                
                # Cache game state setup if possible
                if not hasattr(self, '_cached_game_state') or iteration % 100 == 0:
                    self._cached_game_state = self.setup_game(num_players)
                
                game_state = self._cached_game_state.copy()
                
                # Run CFR iteration
                utilities = self.cfr_solver.train_iteration(game_state)
                
                # Collect statistics less frequently for speed
                if iteration % 5000 == 0:
                    self._collect_training_stats(iteration, utilities, stats)
                
                # Save checkpoint less frequently for speed
                if iteration % (checkpoint_frequency * 2) == 0 and iteration > 0:
                    checkpoint_path = f"data/blueprints/checkpoint_{iteration}.pkl"
                    self.cfr_solver.save_strategy(checkpoint_path)
                    print(f"Saved checkpoint at iteration {iteration}")

        return stats

    def train_blueprint_parallel(self, iterations: int, num_workers: int = 16) -> Dict:
        """Parallel version of train_blueprint using multiple CPU processes.

        Args:
            iterations: Total iterations to train.
            num_workers: Number of worker processes.

        Returns:
            Training statistics similar to train_blueprint.
        """

        num_workers = min(num_workers, mp.cpu_count(), 16)
        if num_workers <= 1:
            return self.train_blueprint(iterations)

        batch_size = max(100, iterations // num_workers)
        total_batches = (iterations + batch_size - 1) // batch_size

        stats = {
            'iterations': [],
            'exploitability': [],
            'total_infosets': [],
            'avg_utility': []
        }

        total_iterations_done = 0
        total_utility = {0: 0.0, 1: 0.0}

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            for batch_id in range(total_batches):
                remaining = iterations - batch_id * batch_size
                current_batch = min(batch_size, remaining)
                args = (batch_id, current_batch, self.config_path)
                futures.append(executor.submit(_worker_train_batch, args))

            for future in as_completed(futures):
                worker_id, batch_done, infosets_data, util = future.result()
                total_iterations_done += batch_done
                for pid, val in util.items():
                    total_utility[pid] += val

                for key, data in infosets_data.items():
                    infoset = self.cfr_solver.get_infoset(key, data['num_actions'])
                    infoset.regret_sum += self.device_config.array(data['regret_sum'])
                    infoset.strategy_sum += self.device_config.array(data['strategy_sum'])
                    infoset.reach_count += data['reach_count']

        self.cfr_solver.iterations += total_iterations_done
        self.cfr_solver.total_utility.update(total_utility)

        exploitability = self.cfr_solver.get_exploitability()
        avg_utility = sum(total_utility.values()) / len(total_utility)

        stats['iterations'].append(total_iterations_done)
        stats['exploitability'].append(exploitability)
        stats['total_infosets'].append(len(self.cfr_solver.infosets))
        stats['avg_utility'].append(avg_utility)

        print(f"Parallel training complete: {total_iterations_done} iterations using {num_workers} workers")

        return stats
    
    def _collect_training_stats(self, iteration: int, utilities: Dict, stats: Dict):
        """Collect and display training statistics"""
        exploitability = self.cfr_solver.get_exploitability()
        avg_utility = sum(utilities.values()) / len(utilities) if utilities else 0.0
        
        stats['iterations'].append(iteration)
        stats['exploitability'].append(exploitability)
        stats['total_infosets'].append(len(self.cfr_solver.infosets))
        stats['avg_utility'].append(avg_utility)
        
        if iteration % 10000 == 0:
            # Get memory info if using GPU
            if self.device_config.use_gpu:
                used_mem, total_mem = self.device_config.get_memory_info()
                mem_usage = f"GPU Memory: {used_mem / 1e9:.1f}GB / {total_mem / 1e9:.1f}GB"
                
                # Additional memory stats for batch processing
                if hasattr(self.cfr_solver, 'get_memory_usage_stats'):
                    mem_stats = self.cfr_solver.get_memory_usage_stats()
                    utilization = mem_stats.get('gpu_utilization', 0)
                    mem_usage += f" ({utilization:.1f}% util)"
            else:
                mem_usage = "CPU Mode"
            
            print(f"Iteration {iteration}: "
                  f"Exploitability={exploitability:.6f}, "
                  f"InfoSets={len(self.cfr_solver.infosets)}, "
                  f"AvgUtility={avg_utility:.2f}, "
                  f"{mem_usage}")
            
            # Additional quality metrics
            if iteration > 0:
                # Strategy stability (Nash distance from previous checkpoint)
                if iteration >= 20000:  # Have previous checkpoint to compare
                    prev_checkpoint = f"data/blueprints/checkpoint_{iteration-10000}.pkl"
                    if os.path.exists(prev_checkpoint):
                        try:
                            from evaluation.metrics import PokerMetrics
                            from cfr.linear_cfr import LinearCFR
                            
                            # Load previous strategy
                            prev_cfr = LinearCFR(self.card_abstraction, self.action_abstraction)
                            prev_cfr.load_strategy(prev_checkpoint)
                            
                            # Compute Nash distance
                            nash_dist = PokerMetrics.nash_distance(self.cfr_solver, prev_cfr)
                            print(f"         Nash distance from prev: {nash_dist:.6f}")
                            
                            # Store in stats
                            if 'nash_distance' not in stats:
                                stats['nash_distance'] = []
                            stats['nash_distance'].append(nash_dist)
                            
                        except Exception as e:
                            print(f"         Could not compute Nash distance: {e}")
            
            # Synchronize GPU operations for accurate timing
            if self.device_config.use_gpu:
                self.device_config.synchronize()
        
        print("Blueprint training complete!")
        return stats
    
    def save_blueprint(self, filepath: str):
        """Save final blueprint strategy"""
        self.cfr_solver.save_strategy(filepath)
        print(f"Blueprint saved to {filepath}")
    
    def evaluate_blueprint(self, num_hands: int = 1000) -> Dict:
        """
        Evaluate blueprint strategy by playing random hands.
        """
        print(f"Evaluating blueprint over {num_hands} hands...")
        
        total_utility = 0
        hands_played = 0
        
        for hand_idx in tqdm(range(num_hands), desc="Evaluating"):
            try:
                game_state = self.setup_game()
                
                # Add safety counter to prevent infinite loops
                max_actions = 1000  # Maximum actions per hand
                action_count = 0
                
                # Play hand using blueprint strategy
                while not game_state.is_terminal() and action_count < max_actions:
                    current_player = game_state.current_player
                    
                    # Get strategy from blueprint
                    infoset_key = self.cfr_solver.create_infoset_key(game_state, current_player)
                    strategy = self.cfr_solver.get_strategy(infoset_key)
                    
                    # Get available actions
                    abstract_actions = self.action_abstraction.get_abstract_actions(
                        game_state, current_player
                    )
                    
                    if not abstract_actions:
                        # Force terminal state if no actions available
                        break
                    
                    # Sample action from strategy
                    if strategy is not None and len(strategy) == len(abstract_actions):
                        action_idx = random.choices(range(len(abstract_actions)), 
                                                   weights=strategy)[0]
                    else:
                        action_idx = random.randint(0, len(abstract_actions) - 1)
                    
                    desc, action_type, amount = abstract_actions[action_idx]
                    
                    # Apply action
                    success = game_state.apply_action(current_player, action_type, amount)
                    action_count += 1
                    
                    if not success:
                        # Force terminal state if action application fails
                        break
                
                # Only collect results if we have a valid terminal state
                if game_state.is_terminal():
                    payoffs = game_state.get_payoffs()
                    total_utility += sum(payoffs)
                    hands_played += 1
                elif action_count >= max_actions:
                    # Handle case where we hit the action limit
                    hands_played += 1  # Still count it but with 0 utility
                else:
                    hands_played += 1
                    
            except Exception as e:
                # Silently handle errors and continue
                hands_played += 1
        
        avg_utility = total_utility / hands_played if hands_played > 0 else 0
        
        evaluation_results = {
            'hands_played': hands_played,
            'average_utility': avg_utility,
            'total_infosets': len(self.cfr_solver.infosets)
        }
        
        print(f"Evaluation complete: {evaluation_results}")
        return evaluation_results