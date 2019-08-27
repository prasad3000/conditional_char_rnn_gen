# conditional_char_rnn_gen
 generating names with conditional character level RNN

# Steps

## Generating referance string
Here we are taking a referance string which include all alphabet(both capital and small) and some symbols. The totle length is 59.

## Read the data
1. The .txt file must be store in data/name/ folder.
2. Read the data in unicode format and convert them to ascii
3. The category is list of all language

## Data Pre-Processing
1. Take a random pair of data.
2. Convert the category into one hot vector of size = number of category
3. Take the char array and generate the onehot vector of size len(char) * 1 * 59
4. Generate the target array which contain the position of that char in the ref string
5. Convert all the np array to tensor

## Model
1. Here we have 3 layer (i2o, i2h, o2o)
2. The input to the i2o is concat of category, input, hidden and the output is fed to o2o
3. The input to the i2h is same as avove and output is fed to o2o
4. The output again fed to the input of next RNN and the hidden is also fed to next hiden recurssively

## Training
1. Initialise the hidden with rand or zeros
2. Traverse the input string letter by letter and pass them to RNN model to get the next output and hidden
3. Do it for the length of the string and add the loss
4. apply the backprop and return the output

## Evaluation
1. Traverse the string letter by letter and pass them into RNN model
2. Sample as a multinomial distribution
3. if the Next char is a EOS then stop and return the generated string
4. Else goto the step one and repeat untill the distance is achieved


## THANK YOU
