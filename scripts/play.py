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
from engine.game_state import GameState, Action
from abstraction.action_abstraction import ActionAbstraction
from engine.hand_evaluator import Card, Rank, Suit
import yaml


class SimpleBot:
    """Simple bot that uses the trained blueprint strategy"""
    
    def __init__(self, blueprint_generator: BlueprintGenerator, player_id: int, debug_mode: bool = False):
        self.blueprint_gen = blueprint_generator
        self.player_id = player_id
        self.cfr_solver = blueprint_generator.cfr_solver
        self.action_abstraction = blueprint_generator.action_abstraction
        self.debug_mode = debug_mode
    
    def get_action(self, game_state: GameState):
        """Get action from bot using blueprint strategy"""
        if game_state.current_player != self.player_id:
            return None
        
        # Get strategy from blueprint
        infoset_key = self.cfr_solver.create_infoset_key(game_state, self.player_id)
        strategy = self.cfr_solver.get_strategy(infoset_key)
        
        # Get available actions
        abstract_actions = self.action_abstraction.get_abstract_actions(
            game_state, self.player_id
        )
        
        if not abstract_actions:
            return None
        
        # Sample action from strategy
        if strategy is not None and len(strategy) == len(abstract_actions):
            # Check if strategy has meaningful weights
            strategy_sum = sum(strategy)
            if strategy_sum > 0:
                action_idx = random.choices(range(len(abstract_actions)), 
                                           weights=strategy)[0]
                if self.debug_mode:
                    print(f"🤖 Using trained strategy: {[f'{p:.3f}' for p in strategy]} -> action {action_idx}")
            else:
                # Strategy exists but is all zeros
                action_idx = random.randint(0, len(abstract_actions) - 1)
                if self.debug_mode:
                    print(f"🎲 Strategy all zeros, using random action {action_idx}")
        else:
            # No strategy found - use conservative heuristic instead of random
            action_idx = self._get_conservative_action(abstract_actions)
            if self.debug_mode:
                print(f"🚫 No strategy found for infoset '{infoset_key}', using conservative action {action_idx}")
                print(f"Available actions: {[desc for desc, _, _ in abstract_actions]}")
        
        desc, action_type, amount = abstract_actions[action_idx]
        return desc, action_type, amount
    
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