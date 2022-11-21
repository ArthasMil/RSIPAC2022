# RSIPAC2022
RSIPAC 2022 @Track3
---
### Track 3: Semantic Segmentations
---

TODO: upload the tuned class weights for CE loss and the DICE loss after the final phase.

---
#### Update 2022/11/20
- We uploaded the final version of our model configuration. We used a cas-training approach: the model is initially trained @512x512 and then tuned @768x768. 

---
#### Update 2022/11/19
- The score is the weighted average F1-Score over all classes, thus the small classes will contribute little to the final score. So we can drop 3-4 classes for simplification. In our solution, we dropped the **photovolts**, the **airports** and the **railway-station**. As the percentage of the **natural-bare-soil** is also very low, we believe this could also be dropped.
- To solve the class imbalance problem, we tuned the class weight manually using the confusion matrix.
- Multi-scales training and inferencing could also improve the performance.


---
#### Update 2022/11/17
### Main features of Basic Model:
- CE + Loss 1:2
- Multiple augmentations with Albu to avoid overfitting
- TTA
- SWA 
- Drop some small samples, such as the  **photovolts**, the **airports** and the **railway-station**. We only classify the objects into 15 classes with the class 0 to be the playground.
---

### WHOLE CODEBASE WILL BE RELEASED AFTER THE COMPETITION.
