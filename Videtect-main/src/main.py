def main():
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    # Initialize video detector
    detector = VideoDetector(WEIGHTS_PATH)
    
    # Extract frames from input video
    print("Extracting frames from video...")
    frames = extract_frames(INPUT_VIDEO, OUTPUT_FOLDER, FRAME_RATE)
    
    # Process each frame
    print("Processing frames...")
    for i, frame in enumerate(frames):
        # Detect objects
        boxes = detector.process_frame(frame)
        
        # Draw boxes on frame
        annotated_frame = detector.draw_boxes(frame.copy(), boxes, LABELS)
        
        # Save processed frame
        output_path = os.path.join(OUTPUT_FOLDER, f'processed_frame_{i:04d}.jpg')
        cv2.imwrite(output_path, annotated_frame)
        
        if i % 10 == 0:
            print(f"Processed {i} frames...")
    
    # Generate output video
    print("Generating output video...")
    generate_video(OUTPUT_FOLDER, 'output.mp4', FRAME_RATE)
    
    print("Processing complete!")

if __name__ == "__main__":
    # Enable memory growth for GPU if available
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    
    main()