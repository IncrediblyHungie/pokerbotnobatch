#!/usr/bin/env python3
"""
Debug script to check if card abstractions are working properly.
"""

import sys
from pathlib import Path

# Add src directory to path
script_dir = Path(__file__).parent
src_dir = script_dir / "src"
sys.path.insert(0, str(src_dir))

from strategy.blueprint_generator import BlueprintGenerator
from engine.game_state import BettingRound

def debug_abstractions():
    print("🔍 DEBUGGING CARD ABSTRACTIONS")
    print("=" * 50)
    
    blueprint_gen = BlueprintGenerator(
        "config/high_quality_training.yaml",
        use_gpu=False,
        use_batch_processing=False
    )
    
    print(f"Config card buckets: {blueprint_gen.config['abstraction']['card_buckets']}")
    print(f"CFR solver type: {type(blueprint_gen.cfr_solver).__name__}")
    
    # Test multiple different games
    unique_keys = set()
    bucket_counts = {"preflop": set(), "flop": set(), "turn": set(), "river": set()}
    
    print("\n🎲 Testing 20 different games...")
    
    for game_num in range(20):
        game_state = blueprint_gen.setup_game(6)
        
        # Test each betting round and player
        for player_id in range(6):
            # Preflop
            game_state.betting_round = BettingRound.PREFLOP
            key = blueprint_gen.cfr_solver.create_infoset_key(game_state, player_id)
            unique_keys.add(key)
            
            # Extract bucket info from key if possible
            if "#" in key:
                parts = key.split("#")
                if len(parts) >= 2:
                    hole_bucket = parts[0]
                    board_bucket = parts[1] 
                    bucket_counts["preflop"].add(hole_bucket)
                    if game_state.betting_round != BettingRound.PREFLOP:
                        bucket_counts[game_state.betting_round.value].add(board_bucket)
            
            if game_num < 3 and player_id == 0:  # Only print first few
                print(f"  Game {game_num+1}, Player {player_id}: {key}")
    
    print(f"\n🎯 ABSTRACTION RESULTS:")
    print(f"Total unique infoset keys: {len(unique_keys)}")
    print(f"Unique preflop buckets seen: {len(bucket_counts['preflop'])}")
    print(f"Unique flop buckets seen: {len(bucket_counts['flop'])}")
    print(f"Unique turn buckets seen: {len(bucket_counts['turn'])}")
    print(f"Unique river buckets seen: {len(bucket_counts['river'])}")
    
    # Test card abstraction directly
    print(f"\n🃏 TESTING CARD ABSTRACTION DIRECTLY:")
    card_abs = blueprint_gen.card_abstraction
    
    # Test a few different hands
    from engine.hand_evaluator import Card, Rank, Suit
    test_hands = [
        [Card(Rank.ACE, Suit.SPADES), Card(Rank.KING, Suit.SPADES)],
        [Card(Rank.TWO, Suit.HEARTS), Card(Rank.THREE, Suit.CLUBS)],
        [Card(Rank.QUEEN, Suit.DIAMONDS), Card(Rank.QUEEN, Suit.HEARTS)],
        [Card(Rank.JACK, Suit.SPADES), Card(Rank.TEN, Suit.SPADES)]
    ]
    
    for i, hand in enumerate(test_hands):
        bucket = card_abs.get_bucket(hand, [], BettingRound.PREFLOP)
        print(f"  Hand {i+1} {[str(c) for c in hand]}: bucket {bucket}")
    
    if len(unique_keys) < 50:
        print(f"\n📋 ALL UNIQUE KEYS:")
        for key in sorted(unique_keys):
            print(f"  {key}")

if __name__ == "__main__":
    debug_abstractions()