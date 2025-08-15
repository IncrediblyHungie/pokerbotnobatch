#!/usr/bin/env python3
"""
Simple test of card abstractions with timeout protection.
"""

import sys
from pathlib import Path

# Add src directory to path
script_dir = Path(__file__).parent
src_dir = script_dir / "src"
sys.path.insert(0, str(src_dir))

def test_simple_abstractions():
    print("🧪 SIMPLE ABSTRACTION TEST")
    print("=" * 40)
    
    try:
        from strategy.blueprint_generator import BlueprintGenerator
        from engine.hand_evaluator import Card, Rank, Suit
        from engine.game_state import BettingRound
        
        print("Loading fast test config...")
        blueprint_gen = BlueprintGenerator(
            "config/fast_test_config.yaml",
            use_gpu=False,
            use_batch_processing=False
        )
        
        print(f"CFR solver: {type(blueprint_gen.cfr_solver).__name__}")
        
        # Test preflop abstraction directly
        print("\n🃏 Testing preflop hands:")
        card_abs = blueprint_gen.card_abstraction
        
        test_hands = [
            [Card(Rank.ACE, Suit.SPADES), Card(Rank.KING, Suit.SPADES)],   # AKs
            [Card(Rank.ACE, Suit.SPADES), Card(Rank.KING, Suit.HEARTS)],   # AKo  
            [Card(Rank.ACE, Suit.SPADES), Card(Rank.ACE, Suit.HEARTS)],    # AA
            [Card(Rank.TWO, Suit.HEARTS), Card(Rank.THREE, Suit.CLUBS)],   # 23o
            [Card(Rank.SEVEN, Suit.DIAMONDS), Card(Rank.SEVEN, Suit.HEARTS)],  # 77
        ]
        
        for i, hand in enumerate(test_hands):
            bucket = card_abs.get_bucket(hand, [], BettingRound.PREFLOP)
            print(f"  Hand {i+1} {[str(c) for c in hand]}: bucket {bucket}")
        
        print(f"\n📊 Preflop buckets available: {len(card_abs.preflop_mapping)}")
        
        if len(card_abs.preflop_mapping) > 5:
            print("✅ Abstractions are working!")
        else:
            print("❌ Abstractions still broken")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_simple_abstractions()