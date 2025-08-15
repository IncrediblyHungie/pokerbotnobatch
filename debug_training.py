#!/usr/bin/env python3
"""
Debug script to understand why so few information sets are being created.
"""

import sys
from pathlib import Path

# Add src directory to path
script_dir = Path(__file__).parent
src_dir = script_dir / "src"
sys.path.insert(0, str(src_dir))

from strategy.blueprint_generator import BlueprintGenerator

def debug_single_iteration():
    """Debug a single training iteration to see what infosets are created"""
    print("🔍 DEBUGGING TRAINING ITERATION")
    print("=" * 50)
    
    # Use proper config
    blueprint_gen = BlueprintGenerator(
        "config/proper_training_config.yaml",
        use_gpu=False,
        use_batch_processing=False
    )
    
    print(f"CFR Solver type: {type(blueprint_gen.cfr_solver).__name__}")
    print(f"Card abstraction buckets: {blueprint_gen.config['abstraction']['card_buckets']}")
    print(f"Action fractions: {blueprint_gen.config['abstraction']['action_fractions']}")
    
    # Debug a few iterations
    for iteration in range(5):
        print(f"\n--- Iteration {iteration + 1} ---")
        
        # Setup 6-player game
        game_state = blueprint_gen.setup_game(6)
        print(f"Game setup: {game_state.num_players} players")
        
        # Show initial hands
        for i, player in enumerate(game_state.players):
            print(f"Player {i}: {player.hole_cards}")
        
        # Count infosets before
        infosets_before = len(blueprint_gen.cfr_solver.infosets)
        
        # Run training iteration
        utilities = blueprint_gen.cfr_solver.train_iteration(game_state)
        
        # Count infosets after
        infosets_after = len(blueprint_gen.cfr_solver.infosets)
        infosets_created = infosets_after - infosets_before
        
        print(f"Infosets before: {infosets_before}")
        print(f"Infosets after: {infosets_after}")
        print(f"New infosets created: {infosets_created}")
        print(f"Utilities: {utilities}")
        
        # Show some infoset keys if any exist
        if blueprint_gen.cfr_solver.infosets:
            print("Sample infoset keys:")
            for i, key in enumerate(list(blueprint_gen.cfr_solver.infosets.keys())[:3]):
                print(f"  {i+1}. {key}")
    
    print(f"\n🎯 FINAL RESULTS:")
    print(f"Total infosets: {len(blueprint_gen.cfr_solver.infosets)}")
    
    if blueprint_gen.cfr_solver.infosets:
        print("\nAll infoset keys:")
        for key in blueprint_gen.cfr_solver.infosets.keys():
            print(f"  {key}")

if __name__ == "__main__":
    debug_single_iteration()