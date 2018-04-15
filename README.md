### Emojify Using LSTM

Adding appropriate emojis to a given sentence. Changing "Congratulations on the promotion! Lets get coffee and talk. Love you!" to "Congratulations on the promotion! 👍 Lets get coffee and talk. ☕️ Love you! ❤️".
We use LSTM model to predict emojis at the end of each sentence in an given input.

### Network Architecture

The following is the network Architecture
<img src="emojifier_nw_arch.png" style="width:700px;height:400px;"> <br>

We calculate the maximim words in sentence and provide embeddings of these words as the input to the LSTM network, propagate through a droput layer and then propapgate thorough another layer. A softmax activation is added at the output layer to precict the emoji

### Results

Training Accuracy = 100%<br>
Test Accuracy = 84%

### Samples outputs

Expected emoji:😞 prediction: are you serious😞<br>
Expected emoji:⚾ prediction: Let us go play baseball	⚾<br>
Expected emoji:😞 prediction: This stupid grader is not working 	😞<br>
Expected emoji:😄 prediction: Congratulation for having a baby	😄<br>
Expected emoji:😞 prediction: stop pissing me off😞<br>
Expected emoji:❤️ prediction: I love taking breaks	❤️<br>
Expected emoji:🍴 prediction: I boiled rice	🍴<br>
