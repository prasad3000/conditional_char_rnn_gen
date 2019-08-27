# conditional_char_rnn_gen
 generating names with conditional character level RNN

# Steps

## Generating referance string
Here we are taking a referance string which include all alphabet(both capital and small) and some symbols. The totle length is 59.

## Read the data
1. The .txt file must be store in data/name/ folder.
2. Read the data in unicode format and convert them to ascii
3. The category is list of all language

## Data Processing
1. Take a random pair of data.
2. Convert the category into one hot vector of size = number of category
3. Take the char array and generate the onehot vector of size len(char) * 1 * 59
4. Generate the target array which contain the position of that char in the ref string
