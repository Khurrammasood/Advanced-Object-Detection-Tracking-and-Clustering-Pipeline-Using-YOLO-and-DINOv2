[10:50 AM, 4/27/2025] Joy boy: Based on this example and your requirements, I'll create an improved version of the code that will:

Detect objects using YOLO (similar to what's shown in your image)
Apply a tracking algorithm to maintain object identity across frames
Crop each detected box (as shown in your example)
Enhance image quality and handle noise for better embedding extraction
Extract embeddings using DINOv2
Cluster the objects based on semantic similarity
Visualize the results effectively

Here's the improved implementation:
[11:50 AM, 4/27/2025] Joy boy: import cv2
import os
import numpy as np
import torch
import time
import logging
from datetime import datetime
from ultralytics import YOLO
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from transformers import AutoImageProcessor, AutoModel, logging as transformers_logging
from PIL import Image

# Suppress transformer warnings
transformers_logging.set_verbosity_error()

# Set up logging and output directory
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_dir = "logs"
output_dir = f"output_{timestamp}"
os.makedirs(log_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(log_dir, f"processing_{timestamp}.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(_name_)

def enhance_crop(crop):
    """Enhance image quality with CLAHE contrast enhancement"""
    if crop is None or crop.size == 0 or crop.shape[0] <= 10 or crop.shape[1] <= 10:
        return None

    try:
        # Apply CLAHE to improve contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        if len(crop.shape) == 3:
            lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            l = clahe.apply(l)
            lab = cv2.merge((l, a, b))
            enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        else:
            enhanced = clahe.apply(crop)

        # Resize to 224x224 for the vision model
        resized = cv2.resize(enhanced, (224, 224), interpolation=cv2.INTER_AREA)
        return resized
    except Exception as e:
        logger.error(f"Error enhancing crop: {str(e)}")
        return None

def load_models(yolo_path):
    """Load YOLO and DINOv2 models"""
    logger.info("Loading models...")

    try:
        # Load YOLO detector with pretrained weights
        yolo_model = YOLO(yolo_path)

        # Load DINOv2 embedding model with progress_bar=False to avoid tqdm output
        model_name = "facebook/dinov2-base"
        processor = AutoImageProcessor.from_pretrained(model_name, use_fast=True)
        vision_model = AutoModel.from_pretrained(model_name)

        # Move vision model to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        vision_model = vision_model.to(device)

        logger.info(f"Models loaded successfully on {device}")
        return yolo_model, vision_model, processor, device

    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        return None, None, None, None

def extract_embedding(model, processor, image, device):
    """Extract embedding from image"""
    try:
        # Convert to RGB for the vision model
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)

        # Process image for the model
        inputs = processor(images=pil_image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Extract features
        with torch.no_grad():
            outputs = model(**inputs)

        # Use CLS token as embedding
        embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        return embedding.flatten()

    except Exception as e:
        logger.error(f"Error extracting embedding: {str(e)}")
        return None

def process_video(video_path, yolo_path, conf_threshold=0.4, sample_rate=5):
    """Process video and collect embeddings"""
    # Load models
    yolo_model, vision_model, processor, device = load_models(yolo_path)
    if None in (yolo_model, vision_model, processor, device):
        return None, None

    # Initialize video capture
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Failed to open video: {video_path}")
        return None, None

    # Get video info
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    logger.info(f"Processing video: {total_frames} frames at {fps} FPS")

    # Initialize object tracker
    tracker = cv2.legacy.TrackerCSRT_create if hasattr(cv2, 'legacy') else cv2.TrackerCSRT_create
    active_trackers = {}
    next_track_id = 0

    # Collections for embeddings and metadata
    embeddings = []
    metadata = []  # (frame_id, object_id, track_id)

    # Processing statistics
    start_time = time.time()
    frame_id = 0

    # Main processing loop
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Process every N frames
        if frame_id % sample_rate == 0:
            try:
                # Detect objects with YOLO
                results = yolo_model(frame, conf=conf_threshold)

                # Update existing trackers
                tracks_to_remove = []
                for track_id, tracker_info in list(active_trackers.items()):
                    tracker_obj = tracker_info['tracker']
                    ok, box = tracker_obj.update(frame)

                    if ok:
                        x, y, w, h = [int(v) for v in box]
                        active_trackers[track_id]['box'] = (x, y, x+w, y+h)
                        active_trackers[track_id]['last_seen'] = frame_id

                        # Only extract new embedding every 10 frames for the same object
                        if frame_id - tracker_info.get('last_embedding_frame', 0) >= 10:
                            # Extract crop and embedding
                            crop = frame[y:y+h, x:x+w].copy()
                            enhanced = enhance_crop(crop)

                            if enhanced is not None:
                                embedding = extract_embedding(vision_model, processor, enhanced, device)

                                if embedding is not None:
                                    embeddings.append(embedding)
                                    metadata.append((frame_id, len(embeddings), track_id))
                                    active_trackers[track_id]['last_embedding_frame'] = frame_id
                    else:
                        # Remove tracker if not seen for a while
                        tracks_to_remove.append(track_id)

                # Remove failed trackers
                for track_id in tracks_to_remove:
                    del active_trackers[track_id]

                # Process new detections and create trackers
                if len(results) > 0 and len(results[0].boxes) > 0:
                    boxes = results[0].boxes.xyxy.cpu().numpy()

                    for i, box in enumerate(boxes):
                        x1, y1, x2, y2 = map(int, box[:4])

                        # Skip if too small or too large
                        width, height = x2-x1, y2-y1
                        if width < 20 or height < 20 or width > frame.shape[1]*0.9:
                            continue

                        # Check if this detection overlaps with any existing tracker
                        new_object = True
                        for track_info in active_trackers.values():
                            tx1, ty1, tx2, ty2 = track_info['box']

                            # Calculate IoU to check overlap
                            ix1, iy1 = max(x1, tx1), max(y1, ty1)
                            ix2, iy2 = min(x2, tx2), min(y2, ty2)
                            iw, ih = max(0, ix2-ix1), max(0, iy2-iy1)
                            intersection = iw * ih
                            union = (x2-x1)(y2-y1) + (tx2-tx1)(ty2-ty1) - intersection
                            iou = intersection / union if union > 0 else 0

                            if iou > 0.5:  # High overlap, same object
                                new_object = False
                                break

                        if new_object:
                            # Create new tracker
                            new_tracker = tracker()
                            ok = new_tracker.init(frame, (x1, y1, x2-x1, y2-y1))

                            if ok:
                                track_id = next_track_id
                                next_track_id += 1

                                active_trackers[track_id] = {
                                    'tracker': new_tracker,
                                    'box': (x1, y1, x2, y2),
                                    'last_seen': frame_id,
                                    'last_embedding_frame': 0  # Force initial embedding
                                }

                                # Process this object
                                crop = frame[y1:y2, x1:x2].copy()
                                enhanced = enhance_crop(crop)

                                if enhanced is not None:
                                    embedding = extract_embedding(vision_model, processor, enhanced, device)

                                    if embedding is not None:
                                        embeddings.append(embedding)
                                        metadata.append((frame_id, len(embeddings), track_id))
                                        active_trackers[track_id]['last_embedding_frame'] = frame_id

                # Log progress
                if frame_id % 50 == 0:
                    elapsed = time.time() - start_time
                    rate = frame_id / elapsed if elapsed > 0 else 0
                    remaining = (total_frames - frame_id) / rate if rate > 0 else 0

                    logger.info(f"Progress: {frame_id}/{total_frames} ({frame_id/total_frames*100:.1f}%) - "
                               f"Speed: {rate:.1f} FPS - Remaining: {remaining/60:.1f} min - "
                               f"Objects: {len(embeddings)}, Tracks: {next_track_id}")

            except Exception as e:
                logger.error(f"Error processing frame {frame_id}: {str(e)}")

        frame_id += 1

    # Cleanup
    cap.release()
    logger.info(f"Processed {frame_id} frames, collected {len(embeddings)} embeddings")

    return embeddings, metadata

def cluster_and_visualize(embeddings, metadata, output_dir, n_clusters=None):
    """Cluster embeddings and visualize results"""
    if not embeddings or len(embeddings) < 2:
        logger.error("Not enough embeddings to cluster")
        return None

    try:
        # Stack embeddings
        X = np.vstack(embeddings)
        logger.info(f"Clustering {X.shape[0]} embeddings of dimension {X.shape[1]}")

        # Determine optimal cluster count if not specified
        if n_clusters is None:
            # Quick auto-determination based on data size
            n_clusters = min(5, max(2, len(embeddings) // 25))

        # Apply KMeans
        logger.info(f"Performing KMeans clustering with {n_clusters} clusters")
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X)

        # Create results with cluster assignments
        results = []
        for i, (frame_id, obj_id, track_id) in enumerate(metadata):
            results.append({
                'frame_id': frame_id,
                'object_id': obj_id,
                'track_id': track_id,
                'cluster': int(cluster_labels[i])
            })

        # Cluster distribution
        cluster_counts = {}
        for label in cluster_labels:
            label_name = f"cluster_{label}"
            cluster_counts[label_name] = cluster_counts.get(label_name, 0) + 1
        logger.info(f"Cluster distribution: {cluster_counts}")

        # Generate t-SNE visualization
        logger.info("Generating t-SNE visualization")
        tsne = TSNE(n_components=2, random_state=42,
                   perplexity=min(30, len(embeddings) - 1))
        reduced = tsne.fit_transform(X)

        # Create plot
        plt.figure(figsize=(12, 10))
        colors = list(mcolors.TABLEAU_COLORS.values())

        # Plot each cluster
        for cluster_id in range(n_clusters):
            mask = cluster_labels == cluster_id
            color = colors[cluster_id % len(colors)]
            plt.scatter(
                reduced[mask, 0], reduced[mask, 1],
                c=[color], label=f'Cluster {cluster_id} ({sum(mask)} items)',
                alpha=0.7
            )

        # Add annotations for select points
        for i, ((x, y), (frame_id, _, track_id)) in enumerate(zip(reduced, metadata)):
            if i % 15 == 0:  # Label every 15th point to reduce clutter
                plt.annotate(f"{track_id}", (x, y), fontsize=8)

        plt.title('Cotton Box Clustering: t-SNE Visualization')
        plt.xlabel('t-SNE component 1')
        plt.ylabel('t-SNE component 2')
        plt.legend(loc='best')
        plt.tight_layout()

        # Save plot
        tsne_path = os.path.join(output_dir, 'clusters_tsne.png')
        plt.savefig(tsne_path, dpi=300)
        plt.close()
        logger.info(f"Visualization saved to {tsne_path}")

        return results

    except Exception as e:
        logger.error(f"Error in clustering: {str(e)}")
        return None

def save_results(results, output_dir):
    """Save clustering results to files"""
    try:
        # Group by cluster
        clusters = {}
        for item in results:
            cluster_id = item['cluster']
            if cluster_id not in clusters:
                clusters[cluster_id] = []
            clusters[cluster_id].append(item)

        # Group by track ID
        tracks = {}
        for item in results:
            track_id = item['track_id']
            if track_id not in tracks:
                tracks[track_id] = []
            tracks[track_id].append(item)

        # Save summary report
        with open(os.path.join(output_dir, 'cluster_summary.txt'), 'w') as f:
            f.write(f"Cotton Box Clustering Analysis\n")
            f.write(f"==========================\n")
            f.write(f"Total objects analyzed: {len(results)}\n")
            f.write(f"Number of clusters: {len(clusters)}\n")
            f.write(f"Number of unique tracks: {len(tracks)}\n\n")

            for cluster_id, items in sorted(clusters.items()):
                f.write(f"CLUSTER {cluster_id} ({len(items)} items):\n")
                # Count occurrences of each track in this cluster
                track_counts = {}
                for item in items:
                    track_id = item['track_id']
                    track_counts[track_id] = track_counts.get(track_id, 0) + 1

                # Sort tracks by frequency
                sorted_tracks = sorted(track_counts.items(), key=lambda x: x[1], reverse=True)

                # List the most common tracks in this cluster
                f.write(f"  Most frequent tracks (ID: count):\n")
                for track_id, count in sorted_tracks[:10]:
                    f.write(f"    Track {track_id}: {count} occurrences\n")

                if len(sorted_tracks) > 10:
                    f.write(f"    ... and {len(sorted_tracks) - 10} more tracks\n")
                f.write("\n")

        # Save CSV for further analysis
        with open(os.path.join(output_dir, 'clusters.csv'), 'w') as f:
            f.write("frame_id,object_id,track_id,cluster\n")
            for item in results:
                f.write(f"{item['frame_id']},{item['object_id']},{item['track_id']},{item['cluster']}\n")

        logger.info(f"Results saved to {output_dir}")
        return True

    except Exception as e:
        logger.error(f"Error saving results: {str(e)}")
        return False

def main(video_path, yolo_path, conf_threshold=0.4, n_clusters=None, sample_rate=5):
    """Main function to process video and cluster cotton boxes"""
    logger.info("Starting cotton box clustering process")
    logger.info(f"Video: {video_path}")
    logger.info(f"YOLO model: {yolo_path}")

    # Process video and collect embeddings
    start_time = time.time()
    embeddings, metadata = process_video(
        video_path,
        yolo_path,
        conf_threshold=conf_threshold,
        sample_rate=sample_rate
    )

    if not embeddings:
        logger.error("Failed to extract embeddings from video")
        return False

    # Cluster the embeddings
    results = cluster_and_visualize(embeddings, metadata, output_dir, n_clusters)

    if not results:
        logger.error("Clustering failed")
        return False

    # Save the results
    success = save_results(results, output_dir)

    # Log completion
    elapsed = time.time() - start_time
    logger.info(f"Process completed in {elapsed:.1f} seconds")
    logger.info(f"Results saved to {output_dir}")

    return success

if _name_ == "_main_":
    # Configuration - update these paths
    video_path = "/content/IMG_2616 (1).mov"  # Update with your video path
    yolo_path = "/content/best (2) (1).pt"    # Update with your YOLO model path

    # Run processing
    success = main(
        video_path=video_path,
        yolo_path=yolo_path,
        conf_threshold=0.4,
        n_clusters=None,  # Auto-determine cluster count
        sample_rate=5     # Process every 5th frame
    )

    if success:
        print(f"✅ Processing complete! Results saved to {output_dir}")
    else:
        print("❌ Processing failed. Check logs for details.")