#!/usr/bin/env python3
"""
Test information set creation to debug why only 5 are being generated.
"""

import sys
from pathlib import Path

# Add src directory to path
script_dir = Path(__file__).parent
src_dir = script_dir / "src"
sys.path.insert(0, str(src_dir))

from strategy.blueprint_generator import BlueprintGenerator

def test_infoset_creation():
    print("🧪 TESTING INFORMATION SET CREATION")
    print("=" * 50)
    
    # Test with 3-player game
    blueprint_gen = BlueprintGenerator(
        "config/working_training_config.yaml",
        use_gpu=False,
        use_batch_processing=False
    )
    
    print(f"CFR Solver: {type(blueprint_gen.cfr_solver).__name__}")
    print(f"Card abstraction config: {blueprint_gen.config['abstraction']['card_buckets']}")
    
    # Test multiple games to see infoset diversity
    all_infoset_keys = set()
    
    for game_num in range(10):
        print(f"\n--- Game {game_num + 1} ---")
        
        game_state = blueprint_gen.setup_game(3)
        
        # Print player hands for debugging
        for i, player in enumerate(game_state.players):
            print(f"Player {i}: {[str(card) for card in player.hole_cards]}")
        
        # Create infoset keys for each player manually
        for player_id in range(3):
            infoset_key = blueprint_gen.cfr_solver.create_infoset_key(game_state, player_id)
            all_infoset_keys.add(infoset_key)
            if game_num < 3:  # Only print first few games
                print(f"  Player {player_id} infoset: {infoset_key}")
        
        # Run one iteration to see if more infosets are created
        before_count = len(blueprint_gen.cfr_solver.infosets)
        utilities = blueprint_gen.cfr_solver.train_iteration(game_state)
        after_count = len(blueprint_gen.cfr_solver.infosets)
        
        if game_num < 3:
            print(f"  Infosets before: {before_count}, after: {after_count}")
    
    print(f"\n🎯 RESULTS:")
    print(f"Total unique infoset keys generated manually: {len(all_infoset_keys)}")
    print(f"Total infosets in CFR solver: {len(blueprint_gen.cfr_solver.infosets)}")
    
    print(f"\nAll infoset keys:")
    for key in sorted(all_infoset_keys):
        print(f"  {key}")
    
    if len(all_infoset_keys) > len(blueprint_gen.cfr_solver.infosets):
        print(f"\n⚠️  Mismatch: {len(all_infoset_keys)} keys generated but only {len(blueprint_gen.cfr_solver.infosets)} stored!")

if __name__ == "__main__":
    test_infoset_creation()