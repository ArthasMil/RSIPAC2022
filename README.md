# RSIPAC2022
RSIPAC 2022 @Track3
---
### Track 3: Semantic Segmentations
---
#### Update 2022/11/19
- The score is the weighted average F1-Score over all classes, thus the small classes will contribute little to the final score. So we can drop 3-4 classes for simplification.
- To solve the class imbalance problem, we tuned the class weight manually using the confusion matrix.
- Multi-scales training and inferencing could also improve the performance.


---
#### Update 2022/11/18
### Main features of Basic Model:
- CE + Loss 1:2
- Multiple augmentations with Albu to avoid overfitting
- TTA
- SWA 

---

### WHOLE CODEBASE WILL BE RELEASED AFTER THE COMPETITION.
