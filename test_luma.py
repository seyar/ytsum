import os
from pathlib import Path
from ytsum import generate_video_segments, generate_video_segments_with_luma, combine_video_segments

# Test script
TEST_SCRIPT = """
NOVA: Welcome to our discussion about artificial intelligence!
ECHO: Today we'll explore how AI is transforming our world.
NOVA: From self-driving cars to medical diagnosis, AI is everywhere.
ECHO: Let's break down the key developments and their impact.
"""

def test_luma_workflow():
    # 1. Generate prompts
    print("Generating prompts...")
    prompts = generate_video_segments(TEST_SCRIPT)
    if prompts:
        print("\nGenerated prompts:")
        for i, prompt in enumerate(prompts, 1):
            print(f"\nPrompt {i}:\n{prompt}")
    else:
        print("Failed to generate prompts")
        return

    # 2. Generate videos
    print("\nGenerating videos...")
    output_dir = Path("test_output")
    output_dir.mkdir(exist_ok=True)
    
    video_paths = generate_video_segments_with_luma(prompts, output_dir)
    if video_paths:
        print("\nGenerated video segments:")
        for path in video_paths:
            print(f"- {path}")
    else:
        print("Failed to generate videos")
        return

    # 3. Combine videos
    print("\nCombining videos...")
    output_path = output_dir / "combined.mp4"
    target_duration = 60  # Test with 60 seconds
    
    if combine_video_segments(video_paths, target_duration, output_path):
        print(f"\nSuccessfully created combined video: {output_path}")
    else:
        print("Failed to combine videos")

if __name__ == "__main__":
    test_luma_workflow() 