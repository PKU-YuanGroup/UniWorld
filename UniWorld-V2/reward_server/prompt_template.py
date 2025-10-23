SCORE_LOGIT = """Here are two images: the original and the edited version. Please evaluate the edited image based on the following editing instruction and requirement. 
Instruction: {prompt}
Requirements: {requirement}
You need to rate the editing result from 0 to 5 based on the accuracy and quality of the edit. 
0: The wrong object was edited, or the edit completely fails to meet the requirements. 
5: The correct object was edited, the requirements were met, and the visual result is high quality.
Response Format (Directly response the score number): 
0-5"""