#!/usr/bin/env python3
"""
Play script for testing the trained Pluribus poker bot.
Allows bot vs bot testing and basic human vs bot play.
"""

import os
import sys
import argparse
import random
from pathlib import Path

# Add src directory to path
script_dir = Path(__file__).parent
src_dir = script_dir.parent / "src"
sys.path.insert(0, str(src_dir))

from strategy.blueprint_generator import BlueprintGenerator
from engine.game_state import GameState, Action, BettingRound
from abstraction.action_abstraction import ActionAbstraction
from engine.hand_evaluator import Card, Rank, Suit
import yaml


class SimpleBot:
    """Pluribus-inspired bot that uses trained blueprint strategy with real-time search"""
    
    def __init__(self, blueprint_generator: BlueprintGenerator, player_id: int, debug_mode: bool = False):
        self.blueprint_gen = blueprint_generator
        self.player_id = player_id
        self.cfr_solver = blueprint_generator.cfr_solver
        self.action_abstraction = blueprint_generator.action_abstraction
        self.debug_mode = debug_mode
        
        # Pluribus-style behavior settings
        self.avoid_limping = True  # Pluribus avoids limping
        self.donk_bet_frequency = 0.15  # Pluribus does more donk betting than humans
        self.bluff_frequency = 0.25  # Reasonable bluffing frequency
        
        # Real-time search parameters (simplified version of Pluribus approach)
        self.search_depth = 3  # Look ahead 3 actions
        self.search_time_limit = 2.0  # 2 seconds per decision (simplified)
    
    def get_action(self, game_state: GameState):
        """Get action from bot using Pluribus-style decision making"""
        if game_state.current_player != self.player_id:
            return None
        
        # Get available actions
        abstract_actions = self.action_abstraction.get_abstract_actions(
            game_state, self.player_id
        )
        
        if not abstract_actions:
            return None
        
        # Apply Pluribus-style real-time search and decision making
        action_idx = self._pluribus_style_decision(game_state, abstract_actions)
        
        desc, action_type, amount = abstract_actions[action_idx]
        return desc, action_type, amount
    
    def _pluribus_style_decision(self, game_state: GameState, abstract_actions):
        """Pluribus-inspired decision making with blueprint + real-time search"""
        # First, get baseline strategy from blueprint
        infoset_key = self.cfr_solver.create_infoset_key(game_state, self.player_id)
        strategy = self.cfr_solver.get_strategy(infoset_key)
        
        # If we have a trained strategy, enhance it with real-time search
        if strategy is not None and len(strategy) == len(abstract_actions) and sum(strategy) > 0:
            # Apply Pluribus behavioral adjustments
            modified_strategy = self._apply_pluribus_adjustments(game_state, abstract_actions, strategy)
            
            # Enhance with limited real-time search (simplified Pluribus approach)
            refined_strategy = self._real_time_search_refinement(game_state, abstract_actions, modified_strategy)
            
            action_idx = random.choices(range(len(abstract_actions)), 
                                       weights=refined_strategy)[0]
            if self.debug_mode:
                print(f"🤖 Blueprint: {[f'{p:.3f}' for p in strategy]}")
                print(f"🎯 Adjusted: {[f'{p:.3f}' for p in modified_strategy]}")
                print(f"🔍 Refined: {[f'{p:.3f}' for p in refined_strategy]} -> action {action_idx}")
            return action_idx
        
        # Fallback to Pluribus-style heuristics with search
        if self.debug_mode:
            print(f"🚫 No strategy found, using Pluribus heuristics with search")
        return self._pluribus_heuristic_action(game_state, abstract_actions)
    
    def _real_time_search_refinement(self, game_state: GameState, abstract_actions, base_strategy):
        """Simplified real-time search to refine strategy (inspired by Pluribus)"""
        # Simplified version: evaluate each action's expected value over short horizon
        action_values = []
        
        for i, (desc, action_type, amount) in enumerate(abstract_actions):
            # Simulate taking this action and estimate value
            expected_value = self._estimate_action_value(game_state, action_type, amount)
            action_values.append(expected_value)
        
        # Blend blueprint strategy with search results
        if max(action_values) > min(action_values):  # If search provides meaningful distinction
            # Normalize action values
            min_val = min(action_values)
            max_val = max(action_values)
            normalized_values = [(v - min_val) / (max_val - min_val) for v in action_values]
            
            # Blend with blueprint (70% blueprint, 30% search)
            blended_strategy = []
            for i in range(len(abstract_actions)):
                blueprint_weight = base_strategy[i] * 0.7
                search_weight = normalized_values[i] * 0.3
                blended_strategy.append(blueprint_weight + search_weight)
            
            # Normalize to sum to 1
            total = sum(blended_strategy)
            if total > 0:
                blended_strategy = [w / total for w in blended_strategy]
                return blended_strategy
        
        # If search doesn't provide value, return original strategy
        return base_strategy
    
    def _estimate_action_value(self, game_state: GameState, action_type: Action, amount: int):
        """Estimate the value of taking an action (simplified)"""
        # Create a copy of game state and simulate the action
        try:
            simulated_state = game_state.copy()
            success = simulated_state.apply_action(self.player_id, action_type, amount)
            
            if not success:
                return -1.0  # Invalid actions have very low value
            
            # Simple heuristic evaluation
            if action_type == Action.FOLD:
                return -0.5  # Folding has negative value but not terrible
            
            # Estimate based on pot odds and hand strength
            hand_strength = self._evaluate_hand_strength(game_state)
            pot_size = simulated_state.pot
            
            if action_type in [Action.RAISE, Action.ALL_IN]:
                # Aggressive actions: value depends on hand strength and pot size
                value = hand_strength * 1.5 - 0.3  # Base aggressive value
                
                # Adjust for pot size (bigger pots = more valuable to win)
                if pot_size > game_state.big_blind * 10:
                    value += 0.2
                    
                return value
                
            elif action_type == Action.CALL:
                # Calling: value based on pot odds vs hand strength
                call_amount = max(0, game_state.current_bet - game_state.players[self.player_id].bet_this_round)
                if call_amount > 0 and pot_size > 0:
                    pot_odds = call_amount / (pot_size + call_amount)
                    return hand_strength - pot_odds  # Simple pot odds calculation
                else:
                    return hand_strength * 0.5
                    
            elif action_type == Action.CHECK:
                return hand_strength * 0.3  # Checking is conservative
                
        except Exception:
            # If simulation fails, return neutral value
            return 0.0
        
        return 0.0
    
    def _apply_pluribus_adjustments(self, game_state: GameState, abstract_actions, base_strategy):
        """Apply Pluribus-style behavioral adjustments to base strategy"""
        adjusted = list(base_strategy.copy())
        
        # 1. Avoid limping (calling big blind preflop)
        if (self.avoid_limping and game_state.betting_round == BettingRound.PREFLOP and 
            game_state.current_bet == game_state.big_blind):
            
            for i, (desc, action_type, amount) in enumerate(abstract_actions):
                if action_type == Action.CALL and "call" in desc.lower():
                    # Reduce limping probability and redistribute to fold/raise
                    limping_reduction = adjusted[i] * 0.7
                    adjusted[i] *= 0.3
                    
                    # Redistribute to aggressive actions
                    for j, (desc2, action_type2, _) in enumerate(abstract_actions):
                        if action_type2 in [Action.RAISE, Action.ALL_IN]:
                            adjusted[j] += limping_reduction / sum(1 for _, at, _ in abstract_actions if at in [Action.RAISE, Action.ALL_IN])
        
        # 2. Increase donk betting frequency
        if (game_state.betting_round != BettingRound.PREFLOP and 
            len(game_state.round_betting_history[game_state.betting_round]) == 0):
            
            for i, (desc, action_type, amount) in enumerate(abstract_actions):
                if action_type in [Action.RAISE] and "bet" in desc.lower():
                    adjusted[i] *= (1.0 + self.donk_bet_frequency)
        
        # Normalize to ensure probabilities sum to 1
        total = sum(adjusted)
        if total > 0:
            adjusted = [p / total for p in adjusted]
        
        return adjusted
    
    def _pluribus_heuristic_action(self, game_state: GameState, abstract_actions):
        """Pluribus-style heuristic when no trained strategy available"""
        # Evaluate hand strength (simplified)
        hand_strength = self._evaluate_hand_strength(game_state)
        
        # Pluribus-style action preferences based on hand strength and situation
        action_scores = []
        
        for i, (desc, action_type, amount) in enumerate(abstract_actions):
            score = self._score_action_pluribus_style(game_state, desc, action_type, amount, hand_strength)
            action_scores.append(score)
        
        # Choose action probabilistically based on scores
        if max(action_scores) > 0:
            # Softmax selection for more realistic play
            import math
            exp_scores = [math.exp(score * 3) for score in action_scores]  # Temperature = 1/3
            total_exp = sum(exp_scores)
            probabilities = [exp_s / total_exp for exp_s in exp_scores]
            return random.choices(range(len(abstract_actions)), weights=probabilities)[0]
        else:
            return random.randint(0, len(abstract_actions) - 1)
    
    def _score_action_pluribus_style(self, game_state, desc, action_type, amount, hand_strength):
        """Score an action using Pluribus-style heuristics"""
        score = 0.0
        
        # Base scoring by action type and hand strength
        if action_type == Action.FOLD:
            score = 1.0 - hand_strength  # Fold more with weak hands
        elif action_type == Action.CHECK:
            score = 0.5  # Neutral action
        elif action_type == Action.CALL:
            # Avoid limping preflop
            if (game_state.betting_round == BettingRound.PREFLOP and 
                game_state.current_bet == game_state.big_blind and self.avoid_limping):
                score = 0.1  # Strongly discourage limping
            else:
                score = hand_strength * 0.8
        elif action_type in [Action.RAISE, Action.ALL_IN]:
            # Aggressive actions
            score = hand_strength * 1.2
            
            # Small raises are preferred over large ones (like Pluribus)
            if "small" in desc.lower() or "25%" in desc.lower() or "33%" in desc.lower():
                score *= 1.2
            elif "large" in desc.lower() or "100%" in desc.lower() or "200%" in desc.lower():
                score *= 0.8
        
        # Position-based adjustments (Pluribus considers position heavily)
        position = (self.player_id - game_state.dealer_position) % game_state.num_players
        if position <= 2:  # Early position - more conservative
            if action_type in [Action.RAISE, Action.ALL_IN]:
                score *= 0.8
        elif position >= game_state.num_players - 2:  # Late position - more aggressive
            if action_type in [Action.RAISE, Action.ALL_IN]:
                score *= 1.2
        
        return max(0, score)
    
    def _evaluate_hand_strength(self, game_state):
        """Simple hand strength evaluation (0.0 to 1.0)"""
        player = game_state.players[self.player_id]
        hole_cards = player.hole_cards
        
        if not hole_cards or len(hole_cards) < 2:
            return 0.3  # Default mediocre strength
        
        # Simple preflop hand strength
        if game_state.betting_round == BettingRound.PREFLOP:
            rank1, rank2 = hole_cards[0].rank.value, hole_cards[1].rank.value
            
            # Pairs
            if rank1 == rank2:
                return min(0.95, 0.5 + rank1 / 28.0)  # Higher pairs = stronger
            
            # High cards
            high_card = max(rank1, rank2)
            if high_card >= 12:  # Queen or better
                return 0.7
            elif high_card >= 10:  # Ten or Jack
                return 0.6
            else:
                return 0.4
        
        # Post-flop: use simple heuristic based on hand type
        all_cards = hole_cards + game_state.community_cards
        if len(all_cards) >= 5:
            # Very simplified strength based on hand ranking
            hand_desc = evaluate_hand(hole_cards, game_state.community_cards)
            
            if "Flush" in hand_desc or "Straight" in hand_desc:
                return 0.9
            elif "Three" in hand_desc or "Full House" in hand_desc:
                return 0.85
            elif "Two Pair" in hand_desc:
                return 0.7
            elif "Pair" in hand_desc:
                return 0.6
            else:
                return 0.4
        
        return 0.5  # Default medium strength
    
    def _get_conservative_action(self, abstract_actions):
        """Conservative strategy when no trained strategy exists"""
        # Prefer check/call over aggressive actions
        action_preferences = {
            'check': 0,    # Most preferred
            'call': 1,     
            'fold': 2,     
            'bet': 3,      # Less preferred
            'raise': 4,    # Least preferred
            'all-in': 5    # Only in desperation
        }
        
        # Find the most conservative legal action
        best_score = float('inf')
        best_idx = 0
        
        for i, (desc, action_type, amount) in enumerate(abstract_actions):
            desc_lower = desc.lower()
            
            # Score based on action type
            score = 10  # Default high score
            for action_name, pref_score in action_preferences.items():
                if action_name in desc_lower:
                    score = pref_score
                    break
            
            # Slightly prefer smaller bet sizes
            if 'bet' in desc_lower or 'raise' in desc_lower:
                if 'small' in desc_lower or '25%' in desc_lower or '33%' in desc_lower:
                    score -= 0.5
                elif 'large' in desc_lower or '100%' in desc_lower or '200%' in desc_lower:
                    score += 1
            
            if score < best_score:
                best_score = score
                best_idx = i
        
        return best_idx


def play_bot_vs_bot(blueprint_path: str, config_path: str, num_hands: int = 100, use_gpu: bool = True):
    """Play bot vs bot games"""
    print(f"Playing {num_hands} hands of bot vs bot...")
    
    # Load blueprint with GPU support
    blueprint_gen = BlueprintGenerator(config_path, use_gpu=use_gpu)
    
    if os.path.exists("data/card_abstractions.pkl"):
        blueprint_gen.card_abstraction.load_abstractions("data/card_abstractions.pkl")
    
    blueprint_gen.cfr_solver.load_strategy(blueprint_path)
    
    # Create bots (no debug output during bot vs bot)
    bots = [SimpleBot(blueprint_gen, i, debug_mode=False) for i in range(6)]
    
    wins = [0] * 6
    total_utility = [0.0] * 6
    
    for hand_num in range(num_hands):
        if hand_num % 10 == 0:
            print(f"Hand {hand_num + 1}/{num_hands}")
        
        # Setup game
        game_state = blueprint_gen.setup_game(num_players=6)
        
        # Play hand
        actions_taken = 0
        max_actions = 1000  # Prevent infinite loops
        
        while not game_state.is_terminal() and actions_taken < max_actions:
            current_player = game_state.current_player
            bot = bots[current_player]
            
            # Get bot action
            bot_action = bot.get_action(game_state)
            if bot_action is None:
                break
            
            desc, action_type, amount = bot_action
            
            # Apply action
            success = game_state.apply_action(current_player, action_type, amount)
            if not success:
                print(f"Invalid action: {desc} by player {current_player}")
                break
            
            actions_taken += 1
        
        # Record results
        if game_state.is_terminal():
            payoffs = game_state.get_payoffs()
            
            for i, payoff in enumerate(payoffs):
                total_utility[i] += payoff
                if payoff > 0:
                    wins[i] += 1
    
    # Print results
    print("\nBot vs Bot Results:")
    print("=" * 30)
    for i in range(6):
        avg_utility = total_utility[i] / num_hands
        win_rate = wins[i] / num_hands * 100
        print(f"Player {i}: {wins[i]} wins ({win_rate:.1f}%), "
              f"Avg utility: {avg_utility:.2f}")


def evaluate_hand(hole_cards, community_cards):
    """Simple hand evaluation - returns best hand description"""
    all_cards = hole_cards + community_cards
    if len(all_cards) < 2:
        return "High Card"
    
    # Simple hand evaluation logic
    ranks = [card.rank.value for card in all_cards]
    suits = [card.suit.value for card in all_cards]
    
    # Count rank frequencies
    rank_counts = {}
    for rank in ranks:
        rank_counts[rank] = rank_counts.get(rank, 0) + 1
    
    count_values = sorted(rank_counts.values(), reverse=True)
    
    # Check for flush
    suit_counts = {}
    for suit in suits:
        suit_counts[suit] = suit_counts.get(suit, 0) + 1
    has_flush = max(suit_counts.values()) >= 5 if suit_counts else False
    
    # Check for straight (including Ace-high and Ace-low)
    unique_ranks = sorted(set(ranks), reverse=True)
    has_straight = False
    
    if len(unique_ranks) >= 5:
        # Check for regular straights
        for i in range(len(unique_ranks) - 4):
            if unique_ranks[i] - unique_ranks[i+4] == 4:
                has_straight = True
                break
        
        # Check for Ace-high straight (10-J-Q-K-A)
        if not has_straight and 14 in ranks:  # Ace present
            if all(rank in ranks for rank in [14, 13, 12, 11, 10]):  # A-K-Q-J-10
                has_straight = True
        
        # Check for Ace-low straight (A-2-3-4-5)  
        if not has_straight and 14 in ranks:  # Ace present
            if all(rank in ranks for rank in [14, 2, 3, 4, 5]):  # A-2-3-4-5
                has_straight = True
    
    # Determine hand type
    if has_straight and has_flush:
        return "Straight Flush!"
    elif count_values[0] == 4:
        return "Four of a Kind!"
    elif count_values[0] == 3 and count_values[1] == 2:
        return "Full House!"
    elif has_flush:
        return "Flush!"
    elif has_straight:
        return "Straight!"
    elif count_values[0] == 3:
        return "Three of a Kind"
    elif count_values[0] == 2 and count_values[1] == 2:
        return "Two Pair"
    elif count_values[0] == 2:
        return "Pair"
    else:
        return "High Card"

def format_cards(cards):
    """Format cards for display"""
    if not cards:
        return "[]"
    return "[" + ", ".join(str(card) for card in cards) + "]"

def get_betting_round_name(betting_round):
    """Get readable betting round name"""
    round_names = {
        "preflop": "PREFLOP",
        "flop": "FLOP", 
        "turn": "TURN",
        "river": "RIVER"
    }
    return round_names.get(betting_round.value.lower(), str(betting_round))

def play_human_vs_bot(blueprint_path: str, config_path: str, use_gpu: bool = True):
    """Enhanced human vs bot interface with persistent stacks"""
    print("🎯 Enhanced Human vs Bot Poker - Session Play")
    print("=" * 50)
    print("Actions: 'f' = fold, 'c' = call/check, 'r [amount]' = raise, 'a' = all-in")
    print("Type 'quit' to exit, 'stats' to see session stats\n")
    
    # Load blueprint with GPU support
    blueprint_gen = BlueprintGenerator(config_path, use_gpu=use_gpu)
    
    if os.path.exists("data/card_abstractions.pkl"):
        blueprint_gen.card_abstraction.load_abstractions("data/card_abstractions.pkl")
    
    blueprint_gen.cfr_solver.load_strategy(blueprint_path)
    
    # Create bot (player 1) with debug mode for human vs bot
    bot = SimpleBot(blueprint_gen, 1, debug_mode=True)
    human_player = 0
    
    # Persistent session tracking
    hand_count = 0
    session_start_stack = 10000
    human_stack = session_start_stack
    bot_stack = session_start_stack
    
    # Session statistics
    hands_won_human = 0
    hands_won_bot = 0
    total_winnings_human = 0
    total_winnings_bot = 0
    biggest_pot = 0
    
    print(f"🎮 Starting session with {session_start_stack} chips each")
    print(f"💡 Type 'stats' anytime to see your session progress")
    
    while True:
        # Check if either player is busted
        if human_stack <= 0:
            print(f"\n💸 You're busted! Bot wins the session!")
            print(f"🤖 Bot total winnings: ${bot_stack - session_start_stack:,.2f}")
            break
        elif bot_stack <= 0:
            print(f"\n🎉 Congratulations! You busted the bot!")
            print(f"💰 Your total winnings: ${human_stack - session_start_stack:,.2f}")
            break
            
        hand_count += 1
        
        # Setup heads-up game with current stacks
        game_state = blueprint_gen.setup_game(num_players=2)
        
        # Set current stacks (but don't let them go below blind amounts)
        min_stack = max(blueprint_gen.game_config['big_blind'] * 2, 100)
        game_state.players[human_player].stack = max(human_stack, min_stack)
        game_state.players[1].stack = max(bot_stack, min_stack)
        
        print(f"\n{'='*60}")
        print(f"🎲 HAND #{hand_count}")
        print(f"💰 Your Stack: ${human_stack:,} | Bot Stack: ${bot_stack:,}")
        if hand_count > 1:
            human_change = human_stack - session_start_stack
            bot_change = bot_stack - session_start_stack
            print(f"📊 Session P&L: You {human_change:+.0f} | Bot {bot_change:+.0f}")
        print(f"{'='*60}")
        
        # Store starting stacks for this hand
        hand_start_human = game_state.players[human_player].stack
        hand_start_bot = game_state.players[1].stack
        
        last_betting_round = None
        
        while not game_state.is_terminal():
            current_player = game_state.current_player
            
            if current_player == human_player:
                # Show game state when betting round changes
                if last_betting_round != game_state.betting_round:
                    last_betting_round = game_state.betting_round
                    
                    print(f"\n🎯 {get_betting_round_name(game_state.betting_round)}")
                    print("-" * 30)
                    
                    # Show your cards
                    your_cards = game_state.players[human_player].hole_cards
                    print(f"🃏 Your Hand: {format_cards(your_cards)}")
                    
                    # Show community cards
                    community = game_state.community_cards
                    if community:
                        print(f"🌍 Community: {format_cards(community)}")
                        # Evaluate current best hand
                        best_hand = evaluate_hand(your_cards, community)
                        print(f"✨ Your Best: {best_hand}")
                    else:
                        print(f"🌍 Community: (no cards yet)")
                    
                    # Show pot and stacks
                    your_stack = game_state.players[human_player].stack
                    bot_stack = game_state.players[1].stack
                    print(f"💰 Pot: {game_state.pot}")
                    print(f"💵 Your Stack: {your_stack} | Bot Stack: {bot_stack}")
                    print()
                
                # Human player action
                legal_actions = game_state.get_legal_actions(current_player)
                print(f"Legal actions: {legal_actions}")
                
                action_input = input("Your action: ").strip().lower()
                
                if action_input == 'quit':
                    return
                elif action_input == 'stats':
                    # Show session statistics
                    print(f"\n📊 SESSION STATISTICS (After {hand_count-1} hands)")
                    print("=" * 40)
                    print(f"💰 Your Stack: ${human_stack:,} (started with ${session_start_stack:,})")
                    print(f"🤖 Bot Stack: ${bot_stack:,} (started with ${session_start_stack:,})")
                    print(f"📈 Your P&L: ${human_stack - session_start_stack:+,}")
                    print(f"📉 Bot P&L: ${bot_stack - session_start_stack:+,}")
                    if hand_count > 1:
                        print(f"🏆 Hands Won: You {hands_won_human} | Bot {hands_won_bot}")
                        win_rate = hands_won_human / (hands_won_human + hands_won_bot) * 100 if (hands_won_human + hands_won_bot) > 0 else 0
                        print(f"📊 Your Win Rate: {win_rate:.1f}%")
                        print(f"💥 Biggest Pot: ${biggest_pot:,}")
                    print("=" * 40)
                    continue
                
                # Parse human input
                if action_input == 'f' or action_input == 'fold':
                    action_type, amount = Action.FOLD, 0
                elif action_input == 'c' or action_input == 'call' or action_input == 'check':
                    # Check or call
                    call_amount = max(0, game_state.current_bet - 
                                    game_state.players[current_player].bet_this_round)
                    if call_amount == 0:
                        action_type, amount = Action.CHECK, 0
                    else:
                        # Ensure call amount doesn't exceed player's stack
                        call_amount = min(call_amount, game_state.players[current_player].stack)
                        action_type, amount = Action.CALL, call_amount
                elif action_input.startswith('r'):
                    # Raise
                    try:
                        parts = action_input.split()
                        if len(parts) > 1:
                            raise_size = int(parts[1])
                        else:
                            raise_size = game_state.big_blind * 2
                        
                        # Calculate total amount needed (call + raise)
                        call_amount = max(0, game_state.current_bet - 
                                        game_state.players[current_player].bet_this_round)
                        amount = call_amount + raise_size
                        action_type = Action.RAISE
                    except ValueError:
                        print("Invalid raise amount")
                        continue
                elif action_input == 'a' or action_input == 'allin' or action_input == 'all-in':
                    # All-in
                    action_type, amount = Action.ALL_IN, game_state.players[current_player].stack
                else:
                    print("Invalid action. Use: 'f' = fold, 'c' = call/check, 'r [amount]' = raise, 'a' = all-in")
                    continue
                
                success = game_state.apply_action(current_player, action_type, amount)
                if not success:
                    print("❌ Invalid action, try again")
                    continue
                
            else:
                # Bot player
                bot_action = bot.get_action(game_state)
                if bot_action is None:
                    break
                
                desc, action_type, amount = bot_action
                print(f"Bot action: {desc}")
                
                success = game_state.apply_action(current_player, action_type, amount)
                if not success:
                    break
        
        # Update stacks and show hand results
        if game_state.is_terminal():
            # Calculate new stacks based on final game state
            final_human_stack = game_state.players[human_player].stack
            final_bot_stack = game_state.players[1].stack
            
            # Calculate hand winnings
            human_hand_result = final_human_stack - hand_start_human
            bot_hand_result = final_bot_stack - hand_start_bot
            
            # Update session stacks
            human_stack = final_human_stack
            bot_stack = final_bot_stack
            
            # Update session statistics
            if human_hand_result > 0:
                hands_won_human += 1
                print(f"🎉 You won ${human_hand_result:,.2f} this hand!")
            else:
                hands_won_bot += 1
                print(f"😞 You lost ${abs(human_hand_result):,.2f} this hand")
                
            # Track biggest pot
            if game_state.pot > biggest_pot:
                biggest_pot = game_state.pot
                
            # Show running totals
            session_profit = human_stack - session_start_stack
            print(f"💰 Your session total: ${session_profit:+,.2f}")
            
            # Show final community cards and hands if reached showdown
            if len(game_state.community_cards) == 5:
                print(f"\n🃏 SHOWDOWN")
                print(f"Your hand: {format_cards(game_state.players[human_player].hole_cards)}")
                print(f"Bot hand: {format_cards(game_state.players[1].hole_cards)}")
                print(f"Board: {format_cards(game_state.community_cards)}")
                
                # Use proper HandEvaluator for accurate results
                from engine.hand_evaluator import HandEvaluator
                evaluator = HandEvaluator()
                
                human_cards = game_state.players[human_player].hole_cards + game_state.community_cards
                bot_cards = game_state.players[1].hole_cards + game_state.community_cards
                
                human_hand_rank, human_kickers = evaluator.evaluate_hand(human_cards)
                bot_hand_rank, bot_kickers = evaluator.evaluate_hand(bot_cards)
                
                print(f"Your best: {human_hand_rank.name}")
                print(f"Bot best: {bot_hand_rank.name}")
                
                # Show actual winner from game logic
                if hasattr(game_state, 'winner') and game_state.winner is not None:
                    if isinstance(game_state.winner, list):
                        print(f"🤝 TIE between players {game_state.winner}")
                    elif game_state.winner == human_player:
                        print(f"🎉 YOU WIN!")
                    else:
                        print(f"🤖 BOT WINS!")
        
        # Auto-continue to next hand (no prompt needed)
        input("\nPress Enter for next hand...")
        continue


def main():
    parser = argparse.ArgumentParser(description="Play with trained poker bot")
    parser.add_argument("--blueprint", default="data/blueprints/final_blueprint.pkl",
                       help="Path to trained blueprint")
    parser.add_argument("--config", default="config/game_config.yaml",
                       help="Path to configuration file")
    parser.add_argument("--mode", choices=["bot_vs_bot", "human_vs_bot"], 
                       default="bot_vs_bot",
                       help="Game mode")
    parser.add_argument("--hands", type=int, default=100,
                       help="Number of hands for bot vs bot")
    parser.add_argument("--cpu", action="store_true",
                       help="Force CPU-only mode (disable GPU)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.blueprint):
        print(f"Blueprint not found: {args.blueprint}")
        print("Please train the bot first using scripts/train.py")
        return
    
    use_gpu = not args.cpu
    
    if args.mode == "bot_vs_bot":
        play_bot_vs_bot(args.blueprint, args.config, args.hands, use_gpu=use_gpu)
    elif args.mode == "human_vs_bot":
        play_human_vs_bot(args.blueprint, args.config, use_gpu=use_gpu)


if __name__ == "__main__":
    main()