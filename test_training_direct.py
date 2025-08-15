#!/usr/bin/env python3
"""
Direct training test to see information set creation.
"""

import sys
from pathlib import Path

# Add src directory to path
script_dir = Path(__file__).parent
src_dir = script_dir / "src"
sys.path.insert(0, str(src_dir))

def test_direct_training():
    print("🎯 DIRECT TRAINING TEST")
    print("=" * 40)
    
    try:
        from strategy.blueprint_generator import BlueprintGenerator
        
        print("Loading configuration...")
        blueprint_gen = BlueprintGenerator(
            "config/fast_test_config.yaml",
            use_gpu=False,
            use_batch_processing=False
        )
        
        print(f"CFR solver: {type(blueprint_gen.cfr_solver).__name__}")
        
        # Run a few training iterations directly
        for i in range(5):
            print(f"\n--- Iteration {i+1} ---")
            
            # Setup game
            game_state = blueprint_gen.setup_game(3)
            
            # Count infosets before
            before = len(blueprint_gen.cfr_solver.infosets)
            
            # Train one iteration  
            utilities = blueprint_gen.cfr_solver.train_iteration(game_state)
            
            # Count infosets after
            after = len(blueprint_gen.cfr_solver.infosets)
            
            print(f"Infosets: {before} -> {after} (+{after-before})")
            print(f"Utilities: {utilities}")
            
            if i == 0 and after > before:
                print("✅ New infosets being created!")
            elif i > 0 and after == before:
                print("⚠️ No new infosets this iteration")
        
        print(f"\n🎯 FINAL RESULT:")
        print(f"Total infosets created: {len(blueprint_gen.cfr_solver.infosets)}")
        
        # Show some sample infosets
        if blueprint_gen.cfr_solver.infosets:
            print("\nSample infoset keys:")
            for key in list(blueprint_gen.cfr_solver.infosets.keys())[:5]:
                print(f"  {key}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_direct_training()