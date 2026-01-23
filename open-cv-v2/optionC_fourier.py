"""
OPTION C: FOURIER DESCRIPTORS
- Describe shape in frequency domain
- Rotation, scale, and translation invariant
- Great for closed contours
- Works well with curves
- Temporal stability included
"""

import cv2
import numpy as np
from pathlib import Path
from collections import deque, Counter

# ============================================================================
# FOURIER DESCRIPTOR EXTRACTOR
# ============================================================================
class FourierDescriptor:
    """
    Fourier Descriptors: represent shape using Fourier transform of contour
    Inherently rotation, scale, and translation invariant
    """
    
    def __init__(self, n_descriptors=20):
        """
        Args:
            n_descriptors: Number of Fourier coefficients to keep
        """
        self.n_descriptors = n_descriptors
    
    def extract_contour(self, image):
        """Extract the main contour from image"""
        contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        if not contours:
            return None
        
        # Get largest contour
        contour = max(contours, key=cv2.contourArea)
        
        # Convert to complex representation
        contour_complex = np.empty(contour.shape[0], dtype=complex)
        contour_complex.real = contour[:, 0, 0]
        contour_complex.imag = contour[:, 0, 1]
        
        return contour_complex
    
    def compute_fourier_descriptors(self, contour_complex):
        """
        Compute Fourier descriptors from complex contour
        
        Returns:
            Normalized Fourier descriptors (rotation, scale, translation invariant)
        """
        if contour_complex is None or len(contour_complex) < 3:
            return None
        
        # Apply FFT
        fourier = np.fft.fft(contour_complex)
        
        # Take only the first n_descriptors coefficients
        descriptors = fourier[:self.n_descriptors]
        
        # Make translation invariant: set DC component to 0
        descriptors[0] = 0
        
        # Make scale invariant: normalize by magnitude of first coefficient
        if np.abs(descriptors[1]) > 0:
            descriptors = descriptors / np.abs(descriptors[1])
        
        # Make rotation invariant: use only magnitudes
        # (Phase information contains rotation)
        descriptors_magnitude = np.abs(descriptors)
        
        return descriptors_magnitude
    
    def compute_descriptor(self, image):
        """Full pipeline: extract contour and compute descriptors"""
        contour = self.extract_contour(image)
        if contour is None:
            return None
        
        descriptors = self.compute_fourier_descriptors(contour)
        return descriptors


# ============================================================================
# FOURIER DESCRIPTOR MATCHER
# ============================================================================
class FourierMatcher:
    """Match shapes using Fourier descriptors"""
    
    def __init__(self, distance_threshold=0.3):
        self.distance_threshold = distance_threshold
    
    def euclidean_distance(self, desc1, desc2):
        """Compute Euclidean distance between descriptors"""
        # Ensure same length
        min_len = min(len(desc1), len(desc2))
        desc1 = desc1[:min_len]
        desc2 = desc2[:min_len]
        
        return np.linalg.norm(desc1 - desc2)
    
    def match(self, query_desc, reference_descs, threshold=None):
        """
        Match query descriptor against reference descriptors
        
        Returns:
            best_match dictionary
        """
        if query_desc is None:
            return None
        
        if threshold is None:
            threshold = self.distance_threshold
        
        best_distance = float('inf')
        best_match = None
        all_distances = []
        
        for ref in reference_descs:
            if ref['descriptors'] is None:
                continue
            
            distance = self.euclidean_distance(query_desc, ref['descriptors'])
            
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
                    'confidence': max(0, 100 * (1 - distance / 2))
                }
        
        # Check threshold
        if best_distance > threshold:
            return {
                'id': -1,
                'name': 'FAKE/UNKNOWN',
                'distance': best_distance,
                'confidence': 0
            }
        
        # Add discrimination info
        sorted_distances = sorted(all_distances, key=lambda x: x['distance'])
        if len(sorted_distances) > 1:
            best_match['second_best'] = sorted_distances[1]['name']
            best_match['second_distance'] = sorted_distances[1]['distance']
            
            if sorted_distances[1]['distance'] > 0:
                best_match['discrimination'] = sorted_distances[1]['distance'] / (best_distance + 0.001)
            else:
                best_match['discrimination'] = 1.0
        
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
    
    descriptor = FourierDescriptor(n_descriptors=20)
    
    for idx, sym_file in enumerate(symbol_files):
        img = cv2.imread(str(sym_file), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        
        # Threshold
        _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        binary = cv2.resize(binary, (256, 256))
        
        # Compute Fourier descriptors
        desc = descriptor.compute_descriptor(binary)
        
        if desc is not None:
            references.append({
                'id': idx,
                'name': sym_file.stem,
                'filename': sym_file.name,
                'descriptors': desc,
                'image': binary
            })
            print(f"  ✓ [{idx}] {sym_file.name} - {len(desc)} Fourier coefficients")
    
    print(f"{'='*70}\n✓ Loaded {len(references)} symbols\n{'='*70}\n")
    return references


# ============================================================================
# MAIN
# ============================================================================
def main():
    refs = load_reference_symbols('sym')
    if not refs:
        return
    
    descriptor = FourierDescriptor(n_descriptors=20)
    matcher = FourierMatcher(distance_threshold=0.3)
    stab = TemporalStabilizer(10, 6)
    
    cap = cv2.VideoCapture(2)
    if not cap.isOpened():
        print("❌ Camera error")
        return
    
    print("🎥 Fourier Descriptor Detection Active\n")
    
    DISTANCE_THRESHOLD = 0.3
    
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
            
            # Compute Fourier descriptors
            query_desc = descriptor.compute_descriptor(roi)
            
            if query_desc is None:
                continue
            
            # Match against references
            match = matcher.match(query_desc, refs, threshold=DISTANCE_THRESHOLD)
            
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
        
        cv2.imshow("Fourier Descriptors", output)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
