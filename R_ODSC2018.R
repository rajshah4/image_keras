##Run this to Install Keras
devtools::install_github("rstudio/keras")
library(keras)
install_keras()
is_keras_available()


######################################
##  Build an Image Classifier
##  https://keras.rstudio.com/articles/tutorial_basic_classification.html
######################################

fashion_mnist <- dataset_fashion_mnist()

c(train_images, train_labels) %<-% fashion_mnist$train
c(test_images, test_labels) %<-% fashion_mnist$test

class_names = c('T-shirt/top',
                'Trouser',
                'Pullover',
                'Dress',
                'Coat', 
                'Sandal',
                'Shirt',
                'Sneaker',
                'Bag',
                'Ankle boot')


library(tidyr)
library(ggplot2)

image_1 <- as.data.frame(train_images[1, , ])
colnames(image_1) <- seq_len(ncol(image_1))
image_1$y <- seq_len(nrow(image_1))
image_1 <- gather(image_1, "x", "value", -y)
image_1$x <- as.integer(image_1$x)

ggplot(image_1, aes(x = x, y = y, fill = value)) +
  geom_tile() +
  scale_fill_gradient(low = "white", high = "black", na.value = NA) +
  scale_y_reverse() +
  theme_minimal() +
  theme(panel.grid = element_blank())   +
  theme(aspect.ratio = 1) +
  xlab("") +
  ylab("")

train_images <- train_images / 255
test_images <- test_images / 255

par(mfcol=c(5,5))
par(mar=c(0, 0, 1.5, 0), xaxs='i', yaxs='i')
for (i in 1:25) { 
  img <- train_images[i, , ]
  img <- t(apply(img, 2, rev)) 
  image(1:28, 1:28, img, col = gray((0:255)/255), xaxt = 'n', yaxt = 'n',
        main = paste(class_names[train_labels[i] + 1]))
}

model <- keras_model_sequential()
model %>%
  layer_flatten(input_shape = c(28, 28)) %>%
  layer_dense(units = 128, activation = 'relu') %>%
  layer_dense(units = 10, activation = 'softmax')

model %>% compile(
  optimizer = 'adam', 
  loss = 'sparse_categorical_crossentropy',
  metrics = c('accuracy')
)

model %>% fit(train_images, train_labels, epochs = 5)

score <- model %>% evaluate(test_images, test_labels)

cat('Test loss:', score$loss, "\n")
cat('Test accuracy:', score$acc, "\n")

predictions <- model %>% predict(test_images)

predictions[1, ]

par(mfcol=c(5,5))
par(mar=c(0, 0, 1.5, 0), xaxs='i', yaxs='i')
for (i in 1:25) { 
  img <- test_images[i, , ]
  img <- t(apply(img, 2, rev)) 
  # subtract 1 as labels go from 0 to 9
  predicted_label <- which.max(predictions[i, ]) - 1
  true_label <- test_labels[i]
  if (predicted_label == true_label) {
    color <- '#008800' 
  } else {
    color <- '#bb0000'
  }
  image(1:28, 1:28, img, col = gray((0:255)/255), xaxt = 'n', yaxt = 'n',
        main = paste0(class_names[predicted_label + 1], " (",
                      class_names[true_label + 1], ")"),
        col.main = color)
}


######################################
##  MNIST CNN Embeddings - layers
##  https://keras.rstudio.com/articles/examples/mnist_cnn_embeddings.html
######################################

# Data Preparation -----------------------------------------------------

batch_size <- 128
num_classes <- 10
epochs <- 4 #12

# Input image dimensions
img_rows <- 28
img_cols <- 28

# The data, shuffled and split between train and test sets
mnist <- dataset_mnist()
x_train <- mnist$train$x[1:5000,,]  #Added subset
y_train <- mnist$train$y[1:5000]  #Added subset
x_test <- mnist$test$x[1:1000,,]  #Added subset
y_test <- mnist$test$y[1:1000]  #Added subset

# Redefine  dimension of train/test inputs
x_train <-
  array_reshape(x_train, c(nrow(x_train), img_rows, img_cols, 1))
x_test <-
  array_reshape(x_test, c(nrow(x_test), img_rows, img_cols, 1))
input_shape <- c(img_rows, img_cols, 1)

# Transform RGB values into [0,1] range
x_train <- x_train / 255
x_test <- x_test / 255

cat('x_train_shape:', dim(x_train), '\n')
cat(nrow(x_train), 'train samples\n')
cat(nrow(x_test), 'test samples\n')



embeddings_dir <- file.path(tempdir(), 'embeddings')
if (!file.exists(embeddings_dir))
  dir.create(embeddings_dir)
embeddings_metadata <- file.path(embeddings_dir, 'metadata.tsv')

# we use the class names from the test set as embeddings_metadata
readr::write_tsv(data.frame(y_test), path = embeddings_metadata, col_names = FALSE) ###FIX IN SCRIPT

tensorboard_callback <- callback_tensorboard(
  log_dir = embeddings_dir,
  batch_size = batch_size,
  embeddings_freq = 1,
  # if missing or NULL all embedding layers will be monitored
  embeddings_layer_names = list('features'),
  # single file for all embedding layers, could also be a named list mapping
  # layer names to file names
  embeddings_metadata = embeddings_metadata,
  # data to be embedded
  embeddings_data = x_test
)

# Define Model -----------------------------------------------------------

# Convert class vectors to binary class matrices
y_train <- to_categorical(y_train, num_classes)
y_test <- to_categorical(y_test, num_classes)

# Define model
model <- keras_model_sequential() %>%
  layer_conv_2d(
    filters = 32,
    kernel_size = c(3, 3),
    activation = 'relu',
    input_shape = input_shape
  ) %>%
  layer_conv_2d(filters = 64,
                kernel_size = c(3, 3),
                activation = 'relu') %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_dropout(rate = 0.25) %>%
  layer_flatten() %>%
  # these are the embeddings (activations) we are going to visualize
  layer_dense(units = 128, activation = 'relu', name = 'features') %>%
  layer_dropout(rate = 0.5) %>%
  layer_dense(units = num_classes, activation = 'softmax')

# Compile model
model %>% compile(
  loss = loss_categorical_crossentropy,
  optimizer = optimizer_adadelta(),
  metrics = c('accuracy')
)


# Launch TensorBoard
#
# As the model is being fit you will be able to view the embedings in the 
# Projector tab. On the left, use "color by label" to see the digits displayed
# in 10 different colors. Hover over a point to see its label.
tensorboard(embeddings_dir)

# Train model
model %>% fit(
  x_train,
  y_train,
  batch_size = batch_size,
  epochs = epochs,
  validation_data = list(x_test, y_test),
  callbacks = list(tensorboard_callback)
)

scores <- model %>% evaluate(x_test, y_test, verbose = 0)

# Output metrics
cat('Test loss:', scores[[1]], '\n')
cat('Test accuracy:', scores[[2]], '\n')


######################################
##  Use Pretrained Network
##  https://keras.rstudio.com/articles/applications.html
######################################


# https://keras.rstudio.com/articles/applications.html
# instantiate the model
model <- application_resnet50(weights = 'imagenet')

# load the image
img_path <- "elephant.jpeg"
img <- image_load(img_path, target_size = c(224,224))
x <- image_to_array(img)

# ensure we have a 4d tensor with single element in the batch dimension,
# the preprocess the input for prediction using resnet50
x <- array_reshape(x, c(1, dim(x)))
x <- imagenet_preprocess_input(x)

# make predictions then decode and print them
preds <- model %>% predict(x)
imagenet_decode_predictions(preds, top = 3)[[1]]



######################################
##  Use Word Embeddings - FastText
######################################

library(keras)
library(purrr)

# Function Definitions ----------------------------------------------------

create_ngram_set <- function(input_list, ngram_value = 2){
  indices <- map(0:(length(input_list) - ngram_value), ~1:ngram_value + .x)
  indices %>%
    map_chr(~input_list[.x] %>% paste(collapse = "|")) %>%
    unique()
}

add_ngram <- function(sequences, token_indice, ngram_range = 2){
  ngrams <- map(
    sequences, 
    create_ngram_set, ngram_value = ngram_range
  )
  
  seqs <- map2(sequences, ngrams, function(x, y){
    tokens <- token_indice$token[token_indice$ngrams %in% y]  
    c(x, tokens)
  })
  
  seqs
}


# Parameters --------------------------------------------------------------

# ngram_range = 2 will add bi-grams features
ngram_range <- 1 #2
max_features <- 2000 #20000
maxlen <- 400
batch_size <- 32
embedding_dims <- 50
epochs <- 5


# Data Preparation --------------------------------------------------------

# Load data
imdb_data <- dataset_imdb(num_words = max_features)

##REDUCE LENGTH FOR TRAINING###

# Train sequences
print(length(imdb_data$train$x))
print(sprintf("Average train sequence length: %f", mean(map_int(imdb_data$train$x, length))))

# Test sequences
print(length(imdb_data$test$x)) 
print(sprintf("Average test sequence length: %f", mean(map_int(imdb_data$test$x, length))))

if(ngram_range > 1) {
  
  # Create set of unique n-gram from the training set.
  ngrams <- imdb_data$train$x %>% 
    map(create_ngram_set) %>%
    unlist() %>%
    unique()
  
  # Dictionary mapping n-gram token to a unique integer
  # Integer values are greater than max_features in order
  # to avoid collision with existing features
  token_indice <- data.frame(
    ngrams = ngrams,
    token  = 1:length(ngrams) + (max_features), 
    stringsAsFactors = FALSE
  )
  
  # max_features is the highest integer that could be found in the dataset
  max_features <- max(token_indice$token) + 1
  
  # Augmenting x_train and x_test with n-grams features
  imdb_data$train$x <- add_ngram(imdb_data$train$x, token_indice, ngram_range)
  imdb_data$test$x <- add_ngram(imdb_data$test$x, token_indice, ngram_range)
}

# Pad sequences
imdb_data$train$x <- pad_sequences(imdb_data$train$x, maxlen = maxlen)
imdb_data$test$x <- pad_sequences(imdb_data$test$x, maxlen = maxlen)


# Model Definition --------------------------------------------------------

model <- keras_model_sequential()

model %>%
  layer_embedding(
    input_dim = max_features, output_dim = embedding_dims, 
    input_length = maxlen
  ) %>%
  layer_global_average_pooling_1d() %>%
  layer_dense(1, activation = "sigmoid")

model %>% compile(
  loss = "binary_crossentropy",
  optimizer = "adam",
  metrics = "accuracy"
)


# Fitting -----------------------------------------------------------------

model %>% fit(
  imdb_data$train$x, imdb_data$train$y, 
  batch_size = batch_size,
  epochs = epochs,
  validation_data = list(imdb_data$test$x, imdb_data$test$y)
)


######################################
## Addition RNN
## https://keras.rstudio.com/articles/examples/addition_rnn.html
######################################


library(keras)
library(stringi)

# Function Definitions ----------------------------------------------------

# Creates the char table and sorts them.
learn_encoding <- function(chars){
  sort(chars)
}

# Encode from a character sequence to a one hot integer representation.
# > encode("22+22", char_table)
# [,1] [,2] [,3] [,4] [,5] [,6] [,7] [,8] [,9] [,10] [,11] [,12]
# 2    0    0    0    0    1    0    0    0    0     0     0     0
# 2    0    0    0    0    1    0    0    0    0     0     0     0
# +    0    1    0    0    0    0    0    0    0     0     0     0
# 2    0    0    0    0    1    0    0    0    0     0     0     0
# 2    0    0    0    0    1    0    0    0    0     0     0     0
encode <- function(char, char_table){
  strsplit(char, "") %>%
    unlist() %>%
    sapply(function(x){
      as.numeric(x == char_table)
    }) %>% 
    t()
}

# Decode the one hot representation/probabilities representation
# to their character output.
decode <- function(x, char_table){
  apply(x,1, function(y){
    char_table[which.max(y)]
  }) %>% paste0(collapse = "")
}

# Returns a list of questions and expected answers.
generate_data <- function(size, digits, invert = TRUE){
  
  max_num <- as.integer(paste0(rep(9, digits), collapse = ""))
  
  # generate integers for both sides of question
  x <- sample(1:max_num, size = size, replace = TRUE)
  y <- sample(1:max_num, size = size, replace = TRUE)
  
  # make left side always smaller than right side
  left_side <- ifelse(x <= y, x, y)
  right_side <- ifelse(x >= y, x, y)
  
  results <- left_side + right_side
  
  # pad with spaces on the right
  questions <- paste0(left_side, "+", right_side)
  questions <- stri_pad(questions, width = 2*digits+1, 
                        side = "right", pad = " ")
  if(invert){
    questions <- stri_reverse(questions)
  }
  # pad with spaces on the left
  results <- stri_pad(results, width = digits + 1, 
                      side = "left", pad = " ")
  
  list(
    questions = questions,
    results = results
  )
}

# Parameters --------------------------------------------------------------

# Parameters for the model and dataset
TRAINING_SIZE <- 50000
DIGITS <- 2

# Maximum length of input is 'int + int' (e.g., '345+678'). Maximum length of
# int is DIGITS
MAXLEN <- DIGITS + 1 + DIGITS

# All the numbers, plus sign and space for padding
charset <- c(0:9, "+", " ")
char_table <- learn_encoding(charset)


# Data Preparation --------------------------------------------------------

# Generate Data
examples <- generate_data(size = TRAINING_SIZE, digits = DIGITS)

# Vectorization
x <- array(0, dim = c(length(examples$questions), MAXLEN, length(char_table)))
y <- array(0, dim = c(length(examples$questions), DIGITS + 1, length(char_table)))

for(i in 1:TRAINING_SIZE){
  x[i,,] <- encode(examples$questions[i], char_table)
  y[i,,] <- encode(examples$results[i], char_table)
}

# Shuffle
indices <- sample(1:TRAINING_SIZE, size = TRAINING_SIZE)
x <- x[indices,,]
y <- y[indices,,]


# Explicitly set apart 10% for validation data that we never train over
split_at <- trunc(TRAINING_SIZE/10)
x_val <- x[1:split_at,,]
y_val <- y[1:split_at,,]
x_train <- x[(split_at + 1):TRAINING_SIZE,,]
y_train <- y[(split_at + 1):TRAINING_SIZE,,]

print('Training Data:')
print(dim(x_train))
print(dim(y_train))

print('Validation Data:')
print(dim(x_val))
print(dim(y_val))


# Training ----------------------------------------------------------------

HIDDEN_SIZE <- 128
BATCH_SIZE <- 128
LAYERS <- 1

# Initialize sequential model
model <- keras_model_sequential() 

model %>%
  # "Encode" the input sequence using an RNN, producing an output of HIDDEN_SIZE.
  # Note: In a situation where your input sequences have a variable length,
  # use input_shape=(None, num_feature).
  layer_lstm(HIDDEN_SIZE, input_shape=c(MAXLEN, length(char_table))) %>%
  # As the decoder RNN's input, repeatedly provide with the last hidden state of
  # RNN for each time step. Repeat 'DIGITS + 1' times as that's the maximum
  # length of output, e.g., when DIGITS=3, max output is 999+999=1998.
  layer_repeat_vector(DIGITS + 1)

# The decoder RNN could be multiple layers stacked or a single layer.
# By setting return_sequences to True, return not only the last output but
# all the outputs so far in the form of (num_samples, timesteps,
# output_dim). This is necessary as TimeDistributed in the below expects
# the first dimension to be the timesteps.
for(i in 1:LAYERS)
  model %>% layer_lstm(HIDDEN_SIZE, return_sequences = TRUE)

model %>% 
  # Apply a dense layer to the every temporal slice of an input. For each of step
  # of the output sequence, decide which character should be chosen.
  time_distributed(layer_dense(units = length(char_table))) %>%
  layer_activation("softmax")

# Compiling the model
model %>% compile(
  loss = "categorical_crossentropy", 
  optimizer = "adam", 
  metrics = "accuracy"
)

# Get the model summary
summary(model)

# Fitting loop
model %>% fit( 
  x = x_train, 
  y = y_train, 
  batch_size = BATCH_SIZE, 
  epochs = 5, #70
  validation_data = list(x_val, y_val)
)

# Predict for a new observation
new_obs <- encode("55+22", char_table) %>%
  array(dim = c(1,5,12))
result <- predict(model, new_obs)
result <- result[1,,]
decode(result, char_table)
