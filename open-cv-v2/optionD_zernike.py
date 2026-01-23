"""
OPTION D: ZERNIKE MOMENTS
- Orthogonal rotation-invariant moments
- Better than Hu moments for complex shapes
- Used in character/symbol recognition systems
- Works well with both curves and lines
- Temporal stability included
"""

import cv2
import numpy as np
from pathlib import Path
from collections import deque, Counter

# ============================================================================
# ZERNIKE MOMENTS CALCULATOR
# ============================================================================
class ZernikeMoments:
    """
    Zernike Moments: orthogonal moments on unit disk
    Rotation invariant, excellent for shape description
    """
    
    def __init__(self, radius=21, degree=12):
        """
        Args:
            radius: Radius of unit disk (must be odd)
            degree: Maximum order of Zernike polynomials
        """
        self.radius = radius
        self.degree = degree
        
        # Precompute Zernike polynomials
        self.zernike_polys = self._precompute_zernike_polynomials()
    
    def _factorial(self, n):
        """Compute factorial"""
        if n <= 1:
            return 1
        return n * self._factorial(n - 1)
    
    def _radial_polynomial(self, rho, n, m):
        """
        Compute radial polynomial R_n^m(rho)
        
        Args:
            rho: Radius values [0, 1]
            n: Order
            m: Repetition
        """
        if (n - abs(m)) % 2 != 0:
            return np.zeros_like(rho)
        
        result = np.zeros_like(rho)
        
        for s in range((n - abs(m)) // 2 + 1):
            numerator = ((-1) ** s) * self._factorial(n - s)
            denominator = (
                self._factorial(s) *
                self._factorial((n + abs(m)) // 2 - s) *
                self._factorial((n - abs(m)) // 2 - s)
            )
            result += (numerator / denominator) * (rho ** (n - 2 * s))
        
        return result
    
    def _precompute_zernike_polynomials(self):
        """Precompute Zernike polynomial values on unit disk"""
        # Create coordinate grid
        x, y = np.meshgrid(
            np.linspace(-1, 1, 2 * self.radius + 1),
            np.linspace(-1, 1, 2 * self.radius + 1)
        )
        
        # Convert to polar
        rho = np.sqrt(x**2 + y**2)
        theta = np.arctan2(y, x)
        
        # Mask for unit disk
        mask = rho <= 1.0
        
        polys = {}
        
        # Compute polynomials for all (n, m) pairs
        for n in range(self.degree + 1):
            for m in range(-n, n + 1, 2 if n % 2 == 0 else 2):
                if abs(m) > n or (n - abs(m)) % 2 != 0:
                    continue
                
                # Radial part
                R = self._radial_polynomial(rho, n, abs(m))
                
                # Angular part
                if m >= 0:
                    V = R * np.cos(m * theta)
                else:
                    V = R * np.sin(abs(m) * theta)
                
                # Apply disk mask
                V = V * mask
                
                polys[(n, m)] = V
        
        return polys
    
    def compute_moments(self, image):
        """
        Compute Zernike moments for an image
        
        Args:
            image: Binary image
        
        Returns:
            moments: Dictionary of Zernike moments (rotation invariant)
        """
        # Resize image to fit in precomputed grid
        h, w = image.shape
        size = 2 * self.radius + 1
        
        if h != size or w != size:
            image = cv2.resize(image, (size, size))
        
        # Normalize image to [0, 1]
        image = image.astype(np.float64) / 255.0
        
        moments = {}
        
        # Compute moments for each polynomial
        for (n, m), poly in self.zernike_polys.items():
            # Zernike moment
            moment = np.sum(image * poly)
            
            # Normalization factor
            moment *= (n + 1) / np.pi
            
            # Store magnitude (rotation invariant)
            moments[(n, m)] = abs(moment)
        
        return moments
    
    def moments_to_vector(self, moments):
        """Convert moments dictionary to feature vector"""
        # Sort by (n, m) for consistency
        sorted_keys = sorted(moments.keys())
        vector = np.array([moments[key] for key in sorted_keys])
        
        # Normalize
        if np.linalg.norm(vector) > 0:
            vector = vector / np.linalg.norm(vector)
        
        return vector


# ============================================================================
# ZERNIKE MATCHER
# ============================================================================
class ZernikeMatcher:
    """Match shapes using Zernike moments"""
    
    def __init__(self, distance_threshold=0.15):
        self.distance_threshold = distance_threshold
    
    def euclidean_distance(self, vec1, vec2):
        """Compute Euclidean distance"""
        min_len = min(len(vec1), len(vec2))
        return np.linalg.norm(vec1[:min_len] - vec2[:min_len])
    
    def match(self, query_vec, reference_vecs, threshold=None):
        """Match query against references"""
        if query_vec is None:
            return None
        
        if threshold is None:
            threshold = self.distance_threshold
        
        best_distance = float('inf')
        best_match = None
        all_distances = []
        
        for ref in reference_vecs:
            if ref['vector'] is None:
                continue
            
            distance = self.euclidean_distance(query_vec, ref['vector'])
            
            all_distances.append({
                'id': ref['id'],
                'name': ref['name'],
                'distance': distance
            })
            
            if distance < best_distance:
                best_distance = distance
                best_match = {
                    'id': ref['id'],
                    'name': ref['name'],
                    'distance': distance,
                    'confidence': max(0, 100 * (1 - distance / 0.5))
                }
        
        # Threshold check
        if best_distance > threshold:
            return {
                'id': -1,
                'name': 'FAKE/UNKNOWN',
                'distance': best_distance,
                'confidence': 0
            }
        
        # Discrimination
        sorted_distances = sorted(all_distances, key=lambda x: x['distance'])
        if len(sorted_distances) > 1:
            best_match['second_best'] = sorted_distances[1]['name']
            best_match['second_distance'] = sorted_distances[1]['distance']
            best_match['discrimination'] = sorted_distances[1]['distance'] / (best_distance + 0.001)
        
        return best_match


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
    
    zernike = ZernikeMoments(radius=21, degree=12)
    
    for idx, sym_file in enumerate(symbol_files):
        img = cv2.imread(str(sym_file), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        
        # Threshold
        _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        binary = cv2.resize(binary, (128, 128))
        
        # Compute Zernike moments
        moments = zernike.compute_moments(binary)
        vector = zernike.moments_to_vector(moments)
        
        if vector is not None:
            references.append({
                'id': idx,
                'name': sym_file.stem,
                'filename': sym_file.name,
                'moments': moments,
                'vector': vector,
                'image': binary
            })
            print(f"  ✓ [{idx}] {sym_file.name} - {len(moments)} Zernike moments")
    
    print(f"{'='*70}\n✓ Loaded {len(references)} symbols\n{'='*70}\n")
    return references


# ============================================================================
# MAIN
# ============================================================================
def main():
    refs = load_reference_symbols('sym')
    if not refs:
        return
    
    zernike = ZernikeMoments(radius=21, degree=12)
    matcher = ZernikeMatcher(distance_threshold=0.15)
    stab = TemporalStabilizer(10, 6)
    
    cap = cv2.VideoCapture(2)
    if not cap.isOpened():
        print("❌ Camera error")
        return
    
    print("🎥 Zernike Moments Detection Active\n")
    
    DISTANCE_THRESHOLD = 0.15
    
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
            roi = cv2.resize(symbol_mask[y:y+h, x:x+w], (128, 128))
            
            # Compute Zernike moments
            moments = zernike.compute_moments(roi)
            query_vec = zernike.moments_to_vector(moments)
            
            if query_vec is None:
                continue
            
            # Match
            match = matcher.match(query_vec, refs, threshold=DISTANCE_THRESHOLD)
            
            if match and match['id'] != -1:
                detections.append(match)
                color = (0, 255, 0)
                label = f"{match['name']}"
                detail = f"Dist: {match['distance']:.3f}, Conf: {match['confidence']:.1f}%"
                
                if 'discrimination' in match:
                    disc_text = f"Discrim: {match['discrimination']:.2f}x"
                else:
                    disc_text = ""
            else:
                color = (0, 0, 255)
                label = "FAKE/UNKNOWN"
                detail = f"Dist: {match['distance']:.3f}" if match else "No match"
                disc_text = ""
            
            cv2.rectangle(output, (x, y), (x+w, y+h), color, 3)
            cv2.putText(output, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(output, detail, (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            if disc_text:
                cv2.putText(output, disc_text, (x, y+h+40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        stable = stab.add_detection(min(detections, key=lambda x: x['distance']) if detections else None)
        
        if stable and stable['id'] != -1:
            text = f"STABLE: {stable['name']} ({stable['votes']}/10, {stable['frames_stable']}f)"
            cv2.rectangle(output, (10, 10), (output.shape[1]-10, 60), (0, 100, 0), -1)
            cv2.putText(output, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow("Zernike Moments", output)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
