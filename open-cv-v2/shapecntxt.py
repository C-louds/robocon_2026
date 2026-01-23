"""
OPTION B: SHAPE CONTEXT DESCRIPTORS
- Specifically designed for shape matching
- Works great with curves and complex shapes
- Rotation invariant
- Uses histogram of relative point positions
- Temporal stability included
"""

from collections import Counter, deque
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.distance import cdist


# ============================================================================
# SHAPE CONTEXT DESCRIPTOR
# ============================================================================
class ShapeContextDescriptor:
    """
    Shape Context: describes shape by histogram of relative positions
    of contour points
    """
    
    def __init__(self, n_points=100, n_bins_r=5, n_bins_theta=12):
        """
        Args:
            n_points: Number of points to sample from contour
            n_bins_r: Number of radial bins (log-polar)
            n_bins_theta: Number of angular bins
        """
        self.n_points = n_points
        self.n_bins_r = n_bins_r
        self.n_bins_theta = n_bins_theta
        self.descriptor_dim = n_bins_r * n_bins_theta
    
    def extract_contour_points(self, image):
        """Extract contour and sample points"""
        # Find contours
        contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        if not contours:
            return None
        
        # Get largest contour
        contour = max(contours, key=cv2.contourArea)
        
        # Sample points evenly along contour
        contour_points = contour.reshape(-1, 2).astype(np.float32)
        
        if len(contour_points) < self.n_points:
            # If contour has fewer points, use all
            return contour_points
        
        # Sample uniformly
        indices = np.linspace(0, len(contour_points) - 1, self.n_points, dtype=int)
        sampled_points = contour_points[indices]
        
        return sampled_points
    
    def compute_shape_context(self, points):
        """
        Compute shape context descriptor for each point
        
        Returns:
            descriptors: (n_points, descriptor_dim) array
        """
        if points is None or len(points) < 2:
            return None
        
        n_points = len(points)
        descriptors = np.zeros((n_points, self.descriptor_dim))
        
        # Define log-polar bins
        # Radial bins (log-spaced)
        r_inner = 0.125  # Inner radius
        r_outer = 2.0    # Outer radius
        r_bins = np.logspace(np.log10(r_inner), np.log10(r_outer), self.n_bins_r + 1)
        
        # Angular bins
        theta_bins = np.linspace(0, 2 * np.pi, self.n_bins_theta + 1)
        
        # For each point, compute descriptor
        for i, point in enumerate(points):
            # Compute relative positions
            relative_pos = points - point
            
            # Convert to polar coordinates
            distances = np.sqrt(np.sum(relative_pos**2, axis=1))
            angles = np.arctan2(relative_pos[:, 1], relative_pos[:, 0]) + np.pi  # [0, 2pi]
            
            # Avoid self-point
            mask = distances > 0
            distances = distances[mask]
            angles = angles[mask]
            
            if len(distances) == 0:
                continue
            
            # Normalize distances
            mean_dist = np.mean(distances) + 1e-6
            distances_norm = distances / mean_dist
            
            # Build histogram
            hist = np.zeros((self.n_bins_r, self.n_bins_theta))
            
            for d, theta in zip(distances_norm, angles):
                # Find radial bin
                r_bin = np.digitize(d, r_bins) - 1
                r_bin = np.clip(r_bin, 0, self.n_bins_r - 1)
                
                # Find angular bin
                theta_bin = np.digitize(theta, theta_bins) - 1
                theta_bin = np.clip(theta_bin, 0, self.n_bins_theta - 1)
                
                hist[r_bin, theta_bin] += 1
            
            # Normalize histogram
            hist = hist / (np.sum(hist) + 1e-6)
            
            # Flatten to descriptor
            descriptors[i] = hist.flatten()
        
        return descriptors
    
    def compute_descriptor(self, image):
        """Full pipeline: extract points and compute descriptors"""
        points = self.extract_contour_points(image)
        if points is None:
            return None, None
        
        descriptors = self.compute_shape_context(points)
        return points, descriptors


# ============================================================================
# SHAPE CONTEXT MATCHER
# ============================================================================
class ShapeContextMatcher:
    """Match shapes using Shape Context descriptors"""
    
    def __init__(self):
        self.chi_square_threshold = 2.0  # Threshold for chi-square distance
    
    def chi_square_distance(self, hist1, hist2):
        """Compute chi-square distance between histograms"""
        return 0.5 * np.sum((hist1 - hist2)**2 / (hist1 + hist2 + 1e-10))
    
    def match_descriptors(self, desc1, desc2):
        """
        Match two sets of shape context descriptors
        
        Returns:
            cost_matrix, matches
        """
        if desc1 is None or desc2 is None:
            return None, []
        
        # Compute cost matrix (chi-square distance between all pairs)
        n1, n2 = len(desc1), len(desc2)
        cost_matrix = np.zeros((n1, n2))
        
        for i in range(n1):
            for j in range(n2):
                cost_matrix[i, j] = self.chi_square_distance(desc1[i], desc2[j])
        
        # Find best matches using Hungarian algorithm (approximate with greedy)
        matches = []
        used_j = set()
        
        for i in range(n1):
            # Find best match for point i
            costs = cost_matrix[i]
            sorted_indices = np.argsort(costs)
            
            for j in sorted_indices:
                if j not in used_j and costs[j] < self.chi_square_threshold:
                    matches.append((i, j, costs[j]))
                    used_j.add(j)
                    break
        
        return cost_matrix, matches
    
    def match_shapes(self, query_points, query_desc, ref_points, ref_desc, threshold=0.5):
        """
        Match query shape against reference shape
        
        Returns:
            match_result dictionary
        """
        if query_desc is None or ref_desc is None:
            return {'matched': False, 'score': 0, 'num_matches': 0}
        
        # Match descriptors
        cost_matrix, matches = self.match_descriptors(query_desc, ref_desc)
        
        if not matches:
            return {'matched': False, 'score': 0, 'num_matches': 0}
        
        # Calculate match quality
        num_matches = len(matches)
        avg_cost = np.mean([cost for _, _, cost in matches])
        match_ratio = num_matches / max(len(query_desc), len(ref_desc))
        
        # Combined score
        score = match_ratio * (1 - avg_cost)
        
        # Decide if matched
        matched = score > threshold and num_matches > 10
        
        return {
            'matched': matched,
            'score': score,
            'num_matches': num_matches,
            'avg_cost': avg_cost,
            'match_ratio': match_ratio,
            'confidence': score * 100
        }


# ============================================================================
# TEMPORAL STABILIZER
# ============================================================================
class TemporalStabilizer:
    def __init__(self, buffer_size=10, min_votes=6):
        self.buffer = deque(maxlen=buffer_size)
        self.min_votes = min_votes
        self.current = None
        self.frames = 0
    
    def add_detection(self, detection):
        self.buffer.append(detection if detection else {'id': -1, 'name': 'NONE'})
        
        if len(self.buffer) < self.min_votes:
            return None
        
        ids = [d['id'] for d in self.buffer]
        most_common_id, count = Counter(ids).most_common(1)[0]
        
        if count >= self.min_votes:
            result = next((d for d in self.buffer if d['id'] == most_common_id), None)
            if result:
                if self.current and self.current['id'] == most_common_id:
                    self.frames += 1
                else:
                    self.frames = 1
                    self.current = result
                
                return {**result, 'votes': count, 'frames_stable': self.frames}
        return None


# ============================================================================
# LOAD REFERENCES
# ============================================================================
def load_reference_symbols(sym_dir='sym'):
    references = []
    symbol_files = sorted(Path(sym_dir).glob('*.png'))
    
    if not symbol_files:
        print(f"❌ No PNG files in {sym_dir}/")
        return []
    
    print(f"\n{'='*70}\nLoading {len(symbol_files)} reference symbols\n{'='*70}")
    
    descriptor = ShapeContextDescriptor(n_points=100, n_bins_r=5, n_bins_theta=12)
    
    for idx, sym_file in enumerate(symbol_files):
        img = cv2.imread(str(sym_file), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        
        # Threshold
        _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        binary = cv2.resize(binary, (256, 256))
        
        # Compute shape context
        points, desc = descriptor.compute_descriptor(binary)
        
        if desc is not None:
            references.append({
                'id': idx, 'name': sym_file.stem, 'filename': sym_file.name,
                'points': points, 'descriptors': desc, 'image': binary
            })
            print(f"  ✓ [{idx}] {sym_file.name} - {len(points)} contour points")
    
    print(f"{'='*70}\n✓ Loaded {len(references)} symbols\n{'='*70}\n")
    return references


# ============================================================================
# MAIN
# ============================================================================
def main():
    refs = load_reference_symbols('sym')
    if not refs:
        return
    
    descriptor = ShapeContextDescriptor(n_points=100)
    matcher = ShapeContextMatcher()
    stab = TemporalStabilizer(10, 6)
    
    cap = cv2.VideoCapture(4)
    if not cap.isOpened():
        print("❌ Camera error")
        return
    
    print("🎥 Shape Context Detection Active\n")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv, np.array([0, 70, 50]), np.array([10, 255, 255]))
        mask2 = cv2.inRange(hsv, np.array([170, 70, 50]), np.array([180, 255, 255]))
        symbol_mask = cv2.bitwise_not(cv2.bitwise_or(mask1, mask2))
        
        kernel = np.ones((3, 3), np.uint8)
        symbol_mask = cv2.morphologyEx(symbol_mask, cv2.MORPH_OPEN, kernel)
        symbol_mask = cv2.morphologyEx(symbol_mask, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(symbol_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        output = frame.copy()
        detections = []
        
        for c in contours:
            if cv2.contourArea(c) < 800:
                continue
            
            x, y, w, h = cv2.boundingRect(c)
            roi = cv2.resize(symbol_mask[y:y+h, x:x+w], (256, 256))
            
            # Compute shape context for query
            query_points, query_desc = descriptor.compute_descriptor(roi)
            
            if query_desc is None:
                continue
            
            # Match against all references
            best = None
            best_score = 0
            
            for ref in refs:
                result = matcher.match_shapes(
                    query_points, query_desc,
                    ref['points'], ref['descriptors'],
                    threshold=0.3
                )
                
                if result['matched'] and result['score'] > best_score:
                    best_score = result['score']
                    best = {
                        'id': ref['id'],
                        'name': ref['name'],
                        'score': result['score'],
                        'num_matches': result['num_matches'],
                        'confidence': result['confidence']
                    }
            
            if best:
                detections.append(best)
                color = (0, 255, 0)
                label = f"{best['name']}"
                detail = f"Score: {best['score']:.3f}, Matches: {best['num_matches']}"
            else:
                color = (0, 0, 255)
                label = "FAKE/UNKNOWN"
                detail = "No shape match"
            
            cv2.rectangle(output, (x, y), (x+w, y+h), color, 3)
            cv2.putText(output, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(output, detail, (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        stable = stab.add_detection(max(detections, key=lambda x: x['score']) if detections else None)
        
        if stable and stable['id'] != -1:
            text = f"STABLE: {stable['name']} ({stable['votes']}/10, {stable['frames_stable']}f)"
            cv2.rectangle(output, (10, 10), (output.shape[1]-10, 60), (0, 100, 0), -1)
            cv2.putText(output, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow("Shape Context", output)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
