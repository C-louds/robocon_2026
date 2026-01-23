"""
OPTION A: ENHANCED ORB FEATURE MATCHING
- Advanced preprocessing (CLAHE, denoising, adaptive thresholding)
- More features (1500+ keypoints)
- Better feature distribution
- RANSAC geometric verification
- Temporal stability with voting
- Adaptive thresholds
- Multi-scale detection
"""

import cv2
import numpy as np
from pathlib import Path
from collections import deque, Counter

# ============================================================================
# ADVANCED PREPROCESSING
# ============================================================================
class AdvancedPreprocessor:
    """Advanced image preprocessing for better feature extraction"""
    
    @staticmethod
    def preprocess(image):
        """Apply advanced preprocessing pipeline"""
        # 1. Denoising
        denoised = cv2.fastNlMeansDenoising(image, None, h=10, templateWindowSize=7, searchWindowSize=21)
        
        # 2. CLAHE
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)
        
        # 3. Bilateral filter
        bilateral = cv2.bilateralFilter(enhanced, 9, 75, 75)
        
        # 4. Adaptive thresholding
        binary = cv2.adaptiveThreshold(
            bilateral, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        # 5. Morphological cleanup
        kernel = np.ones((3, 3), np.uint8)
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
        
        return cleaned


# ============================================================================
# ENHANCED ORB EXTRACTOR
# ============================================================================
class EnhancedORBExtractor:
    """Enhanced ORB with grid-based distribution"""
    
    def __init__(self, nfeatures=1500):
        self.orb = cv2.ORB_create(
            nfeatures=nfeatures, scaleFactor=1.2, nlevels=8,
            edgeThreshold=10, firstLevel=0, WTA_K=2,
            scoreType=cv2.ORB_HARRIS_SCORE, patchSize=31, fastThreshold=20
        )
        self.grid_rows = 4
        self.grid_cols = 4
    
    def extract_features(self, image):
        """Extract features with grid distribution"""
        h, w = image.shape[:2]
        grid_h, grid_w = h // self.grid_rows, w // self.grid_cols
        
        all_kp, all_desc = [], []
        
        for i in range(self.grid_rows):
            for j in range(self.grid_cols):
                y1, y2 = i * grid_h, (i + 1) * grid_h if i < self.grid_rows - 1 else h
                x1, x2 = j * grid_w, (j + 1) * grid_w if j < self.grid_cols - 1 else w
                
                cell = image[y1:y2, x1:x2]
                kp, desc = self.orb.detectAndCompute(cell, None)
                
                if kp:
                    for k in kp:
                        k.pt = (k.pt[0] + x1, k.pt[1] + y1)
                    all_kp.extend(kp)
                    if desc is not None:
                        all_desc.append(desc)
        
        if all_desc:
            return all_kp, np.vstack(all_desc)
        return [], None


# ============================================================================
# RANSAC VERIFIER
# ============================================================================
class RANSACVerifier:
    """Geometric verification using RANSAC"""
    
    @staticmethod
    def verify_matches(kp1, kp2, matches):
        """Verify matches with RANSAC homography"""
        if len(matches) < 4:
            return [], None, 0.0
        
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        
        try:
            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        except:
            return [], None, 0.0
        
        if H is None:
            return [], None, 0.0
        
        inliers = [m for m, inl in zip(matches, mask.ravel()) if inl]
        ratio = len(inliers) / len(matches) if matches else 0
        
        return inliers, H, ratio


# ============================================================================
# ENHANCED MATCHER
# ============================================================================
class EnhancedMatcher:
    """Matcher with ratio test and RANSAC"""
    
    def __init__(self):
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        self.verifier = RANSACVerifier()
    
    def match_features(self, query_kp, query_desc, ref_kp, ref_desc, 
                      ratio_threshold=0.75, min_matches=10):
        """Match with verification"""
        if query_desc is None or ref_desc is None or len(query_desc) == 0:
            return None
        
        try:
            matches = self.bf.knnMatch(query_desc, ref_desc, k=2)
        except:
            return None
        
        # Ratio test
        good = [m for pair in matches if len(pair) == 2 
                for m, n in [pair] if m.distance < ratio_threshold * n.distance]
        
        if len(good) < min_matches:
            return {'inlier_matches': 0, 'passed': False}
        
        # RANSAC
        inliers, H, ratio = self.verifier.verify_matches(query_kp, ref_kp, good)
        
        return {
            'good_matches': len(good),
            'inlier_matches': len(inliers),
            'inlier_ratio': ratio,
            'passed': len(inliers) >= min_matches and ratio > 0.3
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
    
    preprocessor = AdvancedPreprocessor()
    extractor = EnhancedORBExtractor(1500)
    
    for idx, sym_file in enumerate(symbol_files):
        img = cv2.imread(str(sym_file), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        
        processed = preprocessor.preprocess(cv2.resize(img, (256, 256)))
        kp, desc = extractor.extract_features(processed)
        
        if desc is not None and len(kp) > 0:
            references.append({
                'id': idx, 'name': sym_file.stem, 'filename': sym_file.name,
                'keypoints': kp, 'descriptors': desc, 'image': processed
            })
            print(f"  ✓ [{idx}] {sym_file.name} - {len(kp)} keypoints")
    
    print(f"{'='*70}\n✓ Loaded {len(references)} symbols\n{'='*70}\n")
    return references


# ============================================================================
# MAIN
# ============================================================================
def main():
    refs = load_reference_symbols('sym')
    if not refs:
        return
    
    prep = AdvancedPreprocessor()
    ext = EnhancedORBExtractor(1500)
    matcher = EnhancedMatcher()
    stab = TemporalStabilizer(10, 6)
    
    cap = cv2.VideoCapture(2)
    if not cap.isOpened():
        print("❌ Camera error")
        return
    
    print("🎥 Enhanced ORB Active\n")
    
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
            processed = prep.preprocess(roi)
            query_kp, query_desc = ext.extract_features(processed)
            
            if query_desc is None:
                continue
            
            best = None
            best_inliers = 0
            
            for ref in refs:
                result = matcher.match_features(query_kp, query_desc, 
                                               ref['keypoints'], ref['descriptors'])
                if result and result['passed'] and result['inlier_matches'] > best_inliers:
                    best_inliers = result['inlier_matches']
                    best = {'id': ref['id'], 'name': ref['name'], 
                           'inlier_matches': result['inlier_matches'],
                           'inlier_ratio': result['inlier_ratio']}
            
            if best:
                detections.append(best)
                color = (0, 255, 0)
                label = f"{best['name']}"
                detail = f"Inliers: {best['inlier_matches']}, Ratio: {best['inlier_ratio']:.2f}"
            else:
                color = (0, 0, 255)
                label = "FAKE/UNKNOWN"
                detail = "No match"
            
            cv2.rectangle(output, (x, y), (x+w, y+h), color, 3)
            cv2.putText(output, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(output, detail, (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        stable = stab.add_detection(max(detections, key=lambda x: x['inlier_matches']) if detections else None)
        
        if stable and stable['id'] != -1:
            text = f"STABLE: {stable['name']} ({stable['votes']}/10, {stable['frames_stable']}f)"
            cv2.rectangle(output, (10, 10), (output.shape[1]-10, 60), (0, 100, 0), -1)
            cv2.putText(output, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow("Enhanced ORB", output)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
