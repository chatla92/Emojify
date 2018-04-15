### Emojify Using LSTM

Adding appropriate emojis to a given sentence. Changing "Congratulations on the promotion! Lets get coffee and talk. Love you!" to "Congratulations on the promotion! 👍 Lets get coffee and talk. ☕️ Love you! ❤️".
We use LSTM model to predict emojis at the end of each sentence in an given input.

### Network Architecture

The following is the network Architecture
<img src="emojifier_nw_arch.png" style="width:700px;height:400px;"> <br>

We calculate the maximim words in sentence and provide embeddings of these words as the input to the LSTM network, propagate through a droput layer and then propapgate thorough another layer. A softmax activation is added at the output layer to precict the emoji

### Results

Training Accuracy = 100%
Test Accuracy = 84%

### Samples outputs

Expected emoji:😞 prediction: are you serious😞
Expected emoji:⚾ prediction: Let us go play baseball	⚾
Expected emoji:😞 prediction: This stupid grader is not working 	😞
Expected emoji:😄 prediction: Congratulation for having a baby	😄
Expected emoji:😞 prediction: stop pissing me off😞
Expected emoji:❤️ prediction: I love taking breaks	❤️
Expected emoji:🍴 prediction: I boiled rice	🍴