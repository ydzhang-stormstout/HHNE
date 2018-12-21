//  HHNE by Yiding Zhang
//  The hhne.cpp code was built upon the word2vec.c from https://code.google.com/archive/p/word2vec/

//  Modifications Copyright (C) 2018
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.

//  Copyright 2013 Google Inc. All Rights Reserved.
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>

//added
#include <vector>
#include <iostream>
#include <string>

#define MAX_STRING 100
#define EXP_TABLE_SIZE 1000
#define MAX_EXP 9
#define MAX_SENTENCE_LENGTH 1000
#define MAX_CODE_LENGTH 40

const int vocab_hash_size = 100000000;  // Maximum 100 * 0.7 = 70M

typedef double real;

struct vocab_word {
  long long cn; 
  int *point;
  char *word, *code, codelen;
};

char train_file[MAX_STRING], output_file[MAX_STRING];
char save_vocab_file[MAX_STRING], read_vocab_file[MAX_STRING];
struct vocab_word *vocab;
int binary = 0, cbow = 0, debug_mode = 2, window = 5, min_count = 5, num_threads = 12, min_reduce = 1;

int *vocab_hash;
long long vocab_max_size = 1000, vocab_size = 0, layer1_size = 100;
long long train_words = 0, word_count_actual = 0, iter = 5, file_size = 0;
real alpha = 0.025, starting_alpha, sample = 1e-3;
real *syn0, *syn1, *syn1neg, *expTable;
clock_t start;
unsigned long long next_random = 1;
float random_radius = 1;

int hs = 0, negative = 5;
const int table_size = 1e8;
int *table;

void InitUnigramTable() {
  int a, i;
  long long train_words_pow = 0; 
  real d1, power = 0.75;
  table = (int *)malloc(table_size * sizeof(int));
  for (a = 0; a < vocab_size; a++) train_words_pow += pow(vocab[a].cn, power);
  i = 0;
  d1 = pow(vocab[i].cn, power) / (real)train_words_pow;
  for (a = 0; a < table_size; a++) {
    table[a] = i;
    if (a / (real)table_size > d1) {
      i++;
      d1 += pow(vocab[i].cn, power) / (real)train_words_pow;
    }
    if (i >= vocab_size) i = vocab_size - 1;
  }
}

void ReadWord(char *word, FILE *fin) {
  int a = 0, ch; 
  while (!feof(fin)) {
    ch = fgetc(fin);
    if (ch == 13) continue;
    if ((ch == ' ') || (ch == '\t') || (ch == '\n')) {
      if (a > 0) {
        if (ch == '\n') ungetc(ch, fin);
        break;
      }
      if (ch == '\n') {
        strcpy(word, (char *)"</s>");
        return;
      } else continue; 
    }
    word[a] = ch;
    a++;
    if (a >= MAX_STRING - 1) a--;
  }
  word[a] = 0;
}

int GetWordHash(char *word) {
  unsigned long long a, hash = 0;
  for (a = 0; a < strlen(word); a++) hash = hash * 257 + word[a];
  hash = hash % vocab_hash_size;
  return hash;
}

int SearchVocab(char *word) {
  unsigned int hash = GetWordHash(word);
  while (1) {
    if (vocab_hash[hash] == -1) return -1;
    if (!strcmp(word, vocab[vocab_hash[hash]].word)) return vocab_hash[hash];
    hash = (hash + 1) % vocab_hash_size;
  }
  return -1;
}

int ReadWordIndex(FILE *fin) {
  char word[MAX_STRING];
  ReadWord(word, fin);
  if (feof(fin)) return -1;
  return SearchVocab(word);
}

int AddWordToVocab(char *word) {
  unsigned int hash, length = strlen(word) + 1;
  if (length > MAX_STRING) length = MAX_STRING;
  vocab[vocab_size].word = (char *)calloc(length, sizeof(char));
  strcpy(vocab[vocab_size].word, word);
  vocab[vocab_size].cn = 0;
  vocab_size++;
  if (vocab_size + 2 >= vocab_max_size) {
    vocab_max_size += 1000;
    vocab = (struct vocab_word *)realloc(vocab, vocab_max_size * sizeof(struct vocab_word));
  }
  hash = GetWordHash(word);
  while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
  vocab_hash[hash] = vocab_size - 1;
  return vocab_size - 1;
}

int VocabCompare(const void *a, const void *b) {
    return ((struct vocab_word *)b)->cn - ((struct vocab_word *)a)->cn;
}

void SortVocab() {
  int a, size;
  unsigned int hash;
  qsort(&vocab[1], vocab_size - 1, sizeof(struct vocab_word), VocabCompare);
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
  size = vocab_size;
  train_words = 0;
  for (a = 0; a < size; a++) {
    if ((vocab[a].cn < min_count) && (a != 0)) {
      vocab_size--;
      free(vocab[a].word);
    } else {
      hash=GetWordHash(vocab[a].word);
      while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
      vocab_hash[hash] = a;
      train_words += vocab[a].cn;
    }
  }
  vocab = (struct vocab_word *)realloc(vocab, (vocab_size + 1) * sizeof(struct vocab_word));
  for (a = 0; a < vocab_size; a++) {
    vocab[a].code = (char *)calloc(MAX_CODE_LENGTH, sizeof(char));
    vocab[a].point = (int *)calloc(MAX_CODE_LENGTH, sizeof(int));
  }
}

void ReduceVocab() {
  int a, b = 0;
  unsigned int hash;
  for (a = 0; a < vocab_size; a++) if (vocab[a].cn > min_reduce) {
    vocab[b].cn = vocab[a].cn;
    vocab[b].word = vocab[a].word;
    b++;
  } else free(vocab[a].word);
  vocab_size = b;
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
  for (a = 0; a < vocab_size; a++) {
    hash = GetWordHash(vocab[a].word);
    while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
    vocab_hash[hash] = a;
  }
  fflush(stdout);
  min_reduce++;
}

void CreateBinaryTree() {
  long long a, b, i, min1i, min2i, pos1, pos2, point[MAX_CODE_LENGTH];
  char code[MAX_CODE_LENGTH];
  long long *count = (long long *)calloc(vocab_size * 2 + 1, sizeof(long long));
  long long *binary = (long long *)calloc(vocab_size * 2 + 1, sizeof(long long));
  long long *parent_node = (long long *)calloc(vocab_size * 2 + 1, sizeof(long long));
  for (a = 0; a < vocab_size; a++) count[a] = vocab[a].cn;
  for (a = vocab_size; a < vocab_size * 2; a++) count[a] = 1e15;
  pos1 = vocab_size - 1;
  pos2 = vocab_size;
  for (a = 0; a < vocab_size - 1; a++) {
    if (pos1 >= 0) {
      if (count[pos1] < count[pos2]) {
        min1i = pos1;
        pos1--;
      } else {
        min1i = pos2;
        pos2++;
      }
    } else {
      min1i = pos2;
      pos2++;
    }
    if (pos1 >= 0) {
      if (count[pos1] < count[pos2]) {
        min2i = pos1;
        pos1--;
      } else {
        min2i = pos2;
        pos2++;
      }
    } else {
      min2i = pos2;
      pos2++;
    }
    count[vocab_size + a] = count[min1i] + count[min2i];
    parent_node[min1i] = vocab_size + a;
    parent_node[min2i] = vocab_size + a;
    binary[min2i] = 1;
  }
  for (a = 0; a < vocab_size; a++) {
    b = a;
    i = 0;
    while (1) {
      code[i] = binary[b];
      point[i] = b;
      i++;
      b = parent_node[b];
      if (b == vocab_size * 2 - 2) break;
    }
    vocab[a].codelen = i;
    vocab[a].point[0] = vocab_size - 2;
    for (b = 0; b < i; b++) {
      vocab[a].code[i - b - 1] = code[b];
      vocab[a].point[i - b] = point[b] - vocab_size;
    }
  }
  free(count);
  free(binary);
  free(parent_node);
}



void LearnVocabFromTrainFile() {
  char word[MAX_STRING];
  FILE *fin;
  long long a, i;
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
  fin = fopen(train_file, "rb");
  if (fin == NULL) {
    printf("ERROR: training data file not found!\n");
    exit(1);
  }
  vocab_size = 0;
  AddWordToVocab((char *)"</s>");
  while (1) {
    ReadWord(word, fin);
    if (feof(fin)) break;
    train_words++;
    if ((debug_mode > 1) && (train_words % 100000 == 0)) {
      printf("%lldK%c", train_words / 1000, 13);
      fflush(stdout);
    }
    i = SearchVocab(word);
    if (i == -1) {
      a = AddWordToVocab(word);
      vocab[a].cn = 1;
    } else vocab[i].cn++;
    if (vocab_size > vocab_hash_size * 0.7) ReduceVocab();
  }
  SortVocab();
  if (debug_mode > 0) {
    printf("Vocab size: %lld\n", vocab_size);
    printf("Words in train file: %lld\n", train_words);
  }
  file_size = ftell(fin);
  fclose(fin);
}

void SaveVocab() {
  long long i;
  FILE *fo = fopen(save_vocab_file, "wb");
  for (i = 0; i < vocab_size; i++) fprintf(fo, "%s %lld\n", vocab[i].word, vocab[i].cn);
  fclose(fo);
}

void ReadVocab() {
  long long a, i = 0;
  char c;
  char word[MAX_STRING];
  FILE *fin = fopen(read_vocab_file, "rb");
  if (fin == NULL) {
    printf("Vocabulary file not found\n");
    exit(1);
  }
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
  vocab_size = 0;
  while (1) {
    ReadWord(word, fin);
    if (feof(fin)) break;
    a = AddWordToVocab(word);
    fscanf(fin, "%lld%c", &vocab[a].cn, &c);
    i++;
  }
  SortVocab();
  if (debug_mode > 0) {
    printf("Vocab size: %lld\n", vocab_size);
    printf("Words in train file: %lld\n", train_words);
  }
  fin = fopen(train_file, "rb");
  if (fin == NULL) {
    printf("ERROR: training data file not found!\n");
    exit(1);
  }
  fseek(fin, 0, SEEK_END);
  file_size = ftell(fin);
  fclose(fin);
}

real* ExponentialMap(real *x, real *v) {
	real x_tmp_norm = 0, v_tmp_norm = 0;
    int c = 0;
	real* tmp_x = (real*)calloc(layer1_size, sizeof(real));
	real* tmp_v = (real*)calloc(layer1_size, sizeof(real));
	for (c = 0; c < layer1_size; c++) {tmp_x[c] = x[c]; tmp_v[c] = v[c];}
    for (c = 0; c < layer1_size; c++) {x_tmp_norm += x[c] * x[c]; v_tmp_norm += v[c] * v[c];}
    x_tmp_norm = sqrt(x_tmp_norm); v_tmp_norm = sqrt(v_tmp_norm);
	real a = 0.98, sqrt_a = sqrt(a);
	if (x_tmp_norm > a) for (c = 0; c < layer1_size; c++) x[c] = sqrt_a * x[c] / x_tmp_norm;
	if (v_tmp_norm > a) for (c = 0; c < layer1_size; c++) v[c] = sqrt_a * v[c] / v_tmp_norm;
    real lambda_x = 0;
    real* map_vec = (real*)calloc(layer1_size, sizeof(real));
    real* normal_tmp_vec = (real*)calloc(layer1_size, sizeof(real));
    real tmp_cof = 0, v_norm = 0, xv_dot = 0;
    for (c = 0; c < layer1_size; c++) lambda_x += x[c] * x[c];
    lambda_x = 2 / (1 - lambda_x);
    for (c = 0; c < layer1_size; c++) v_norm += v[c] * v[c];
    if (v_norm == 0.0) {printf("v_norm = 0\n"); v_norm = 1.0;}
    v_norm = sqrt(v_norm);
    for (c = 0; c < layer1_size; c++) xv_dot += x[c] * v[c] / v_norm;
    tmp_cof = lambda_x * (cosh(lambda_x * v_norm) + xv_dot * sinh(lambda_x * v_norm));
    for (c = 0; c < layer1_size; c++) map_vec[c] = x[c] * tmp_cof;
    tmp_cof = sinh(lambda_x * v_norm) / v_norm;
    for (c = 0; c < layer1_size; c++) map_vec[c] += v[c] * tmp_cof;
    tmp_cof = 1 + (lambda_x - 1) * cosh(lambda_x * v_norm) + lambda_x * xv_dot * sinh(lambda_x * v_norm);
    for (c = 0; c < layer1_size; c++) map_vec[c] = map_vec[c] / tmp_cof;
    tmp_cof = 0;
    for (c = 0; c < layer1_size; c++) tmp_cof += map_vec[c] * map_vec[c];
    tmp_cof = sqrt(tmp_cof);
    if (tmp_cof >= 1 && tmp_cof < 1.01){
    	for (c = 0; c < layer1_size; c++) map_vec[c] = sqrt_a * map_vec[c] / tmp_cof;
    }
    free(normal_tmp_vec);
    free(tmp_x);
    free(tmp_v);
    return map_vec;
}

void InitNet() {
  long long a, b;
  real init_norm = 0;
  a = posix_memalign((void **)&syn0, 128, (long long)vocab_size * layer1_size * sizeof(real));
  if (syn0 == NULL) {printf("Memory allocation failed\n"); exit(1);}
  if (negative>0) {
    a = posix_memalign((void **)&syn1neg, 128, (long long)vocab_size * layer1_size * sizeof(real));
    if (syn1neg == NULL) {printf("Memory allocation failed\n"); exit(1);}
    for (a = 0; a < vocab_size; a++) for (b = 0; b < layer1_size; b++)
     syn1neg[a * layer1_size + b] = 0;
  }
  for (a = 0; a < vocab_size; a++) {
  	for (b = 0; b < layer1_size; b++) {
      next_random = next_random * (unsigned long long)25214903917 + 11;
      syn0[a * layer1_size + b] = (((next_random & 0xFFFF) / (real)65536) - 0.5) / layer1_size * random_radius;
	}
  }
  CreateBinaryTree();
}

void *TrainModelThread(void *id) {
  long long a, b, d, word, last_word, sentence_length = 0, sentence_position = 0;
  long long word_count = 0, last_word_count = 0, sen[MAX_SENTENCE_LENGTH + 1];
  long long l1, l2, c, target, label, local_iter = iter;
  unsigned long long next_random = (long long)id;
  real g;
  real hybolic_alpha, hybolic_beta, hybolic_gamma;
  real hybolic_syn0_norm, hybolic_syn1neg_norm, hybolic_sub_norm, hybolic_distance;
  real deriv_syn0_tmp_cof, deriv_syn1neg_tmp_cof;
  real *deriv_syn0_tmp_arr = (real *)calloc(layer1_size, sizeof(real));
  real *deriv_syn1neg_tmp_arr = (real *)calloc(layer1_size, sizeof(real));
  real *deriv_syn0_poincare = (real *)calloc(layer1_size, sizeof(real));
  real *deriv_syn1neg_poincare = (real *)calloc(layer1_size, sizeof(real));
  real *map_arr;
  real *map_tmp_arr = (real *)calloc(layer1_size, sizeof(real));
  if (map_tmp_arr == NULL) {printf("Memory allocation failed\n"); exit(1);}
  clock_t now;
  real *neu1 = (real *)calloc(layer1_size, sizeof(real));
  real *neu1e = (real *)calloc(layer1_size, sizeof(real));
  FILE *fi = fopen(train_file, "rb");
  fseek(fi, file_size / (long long)num_threads * (long long)id, SEEK_SET);
  while (1) {
    if (word_count - last_word_count > 10000) {
      word_count_actual += word_count - last_word_count;
      last_word_count = word_count;
      if ((debug_mode > 1)) {
        now=clock();
        printf("%cAlpha: %f  Progress: %.2f%%  Words/thread/sec: %.2fk", 13, alpha,
         word_count_actual / (real)(iter * train_words + 1) * 100,
         word_count_actual / ((real)(now - start + 1) / (real)CLOCKS_PER_SEC * 1000));
        fflush(stdout);
      }
      alpha = starting_alpha * (1 - word_count_actual / (real)(iter * train_words + 1));
      if (alpha < starting_alpha * 0.0001) alpha = starting_alpha * 0.0001;
    }
    if (sentence_length == 0) {
      while (1) {
        word = ReadWordIndex(fi);
        if (feof(fi)) break;
        if (word == -1) continue;
        word_count++;
        if (word == 0) break;
        if (sample > 0) {
          real ran = (sqrt(vocab[word].cn / (sample * train_words)) + 1) * (sample * train_words) / vocab[word].cn;
          next_random = next_random * (unsigned long long)25214903917 + 11;
          if (ran < (next_random & 0xFFFF) / (real)65536) continue;
        }
        sen[sentence_length] = word;
        sentence_length++;
        if (sentence_length >= MAX_SENTENCE_LENGTH) break;
      }
      sentence_position = 0;
    }
    if (feof(fi) || (word_count > train_words / num_threads)) {
      word_count_actual += word_count - last_word_count;
      local_iter--;
      if (local_iter == 0) {
      	break;}
      word_count = 0;
      last_word_count = 0;
      sentence_length = 0;
      fseek(fi, file_size / (long long)num_threads * (long long)id, SEEK_SET);
      continue;
    }
    word = sen[sentence_position];
    if (word == -1) continue;
    for (c = 0; c < layer1_size; c++) neu1[c] = 0;
    for (c = 0; c < layer1_size; c++) neu1e[c] = 0;

    next_random = next_random * (unsigned long long)25214903917 + 11;
    b = 0;
    for (a = b; a < window * 2 + 1 - b; a++) if (a != window) {
      c = sentence_position - window + a;
      if (c < 0) continue;
      if (c >= sentence_length) continue;
      last_word = sen[c];
      if (last_word == -1) continue;
      l1 = last_word * layer1_size;
      for (c = 0; c < layer1_size; c++) neu1e[c] = 0;
      if (negative > 0) for (d = 0; d < negative + 1; d++) {
        if (d == 0) {
          target = word;
          label = 1;
        } else {
          next_random = next_random * (unsigned long long)25214903917 + 11;  
	      target = table[(next_random >> 16) % table_size]; 
	      if (target == 0) target = next_random % (vocab_size - 1) + 1;
	      if (target == word) continue;
          label = 0;
        }
        l2 = target * layer1_size;
        hybolic_syn0_norm = 0;
        hybolic_syn1neg_norm = 0;
        hybolic_sub_norm = 0;
        hybolic_distance = 0;
        deriv_syn0_tmp_cof = 0;
        deriv_syn1neg_tmp_cof = 0;
        for (c = 0; c < layer1_size; c++) hybolic_syn0_norm += syn0[c + l1] * syn0[c + l1];
        hybolic_alpha = 1 - hybolic_syn0_norm;
        if (hybolic_alpha < 0.0001) hybolic_alpha = 0.01;
    	if (hybolic_syn0_norm >= 1.01) {printf("error: 2 - syn0_norm is %lf\n", hybolic_syn0_norm); for (c = 0; c < layer1_size; c++) printf("%f ", syn0[c + l1]); exit(1);}
        for (c = 0; c < layer1_size; c++) hybolic_syn1neg_norm += syn1neg[c + l2] * syn1neg[c + l2];
	    if (hybolic_syn1neg_norm >= 1.01) {printf("error: 2 - syn1neg_norm is %lf\n", hybolic_syn1neg_norm); exit(1);}
        hybolic_beta = 1 - hybolic_syn1neg_norm;
        if (hybolic_beta < 0.0001) hybolic_beta = 0.01;
        for (c = 0; c < layer1_size; c++) hybolic_sub_norm += pow((syn0[c + l1] - syn1neg[c + l2]),2);
        hybolic_gamma = 1 + 2 * hybolic_sub_norm / hybolic_alpha / hybolic_beta;
        hybolic_distance = log(hybolic_gamma + sqrt(hybolic_gamma * hybolic_gamma - 1));

        if (0 - hybolic_distance > MAX_EXP) g = (label - 1) * alpha;
        else if (0 - hybolic_distance < -MAX_EXP) g = (label - 0) * alpha;
        else g = (label - expTable[(int)((0 - hybolic_distance + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha;
        for (c = 0; c < layer1_size; c++) deriv_syn0_tmp_cof += syn0[c + l1] * syn1neg[c + l2];
        deriv_syn0_tmp_cof = (hybolic_syn1neg_norm - 2 * deriv_syn0_tmp_cof + 1) / hybolic_alpha / hybolic_alpha;
        for (c = 0; c < layer1_size; c++) deriv_syn0_tmp_arr[c] = syn0[c + l1] * deriv_syn0_tmp_cof;
        for (c = 0; c < layer1_size; c++) deriv_syn1neg_tmp_arr[c] = syn1neg[c + l2] / hybolic_alpha;
        if (hybolic_gamma * hybolic_gamma - 1 < 0.00001) hybolic_gamma += 0.001;
        deriv_syn0_tmp_cof = g * 4 / hybolic_beta / sqrt(hybolic_gamma * hybolic_gamma - 1);
        for (c = 0; c < layer1_size; c++) deriv_syn0_poincare[c] = deriv_syn0_tmp_cof * (deriv_syn1neg_tmp_arr[c] - deriv_syn0_tmp_arr[c]);
        if (deriv_syn0_poincare[0] == 0){
        	for (c = 0; c < layer1_size; c++) if (deriv_syn0_poincare[c] == 0) deriv_syn0_poincare[c] = 0.00001;
        }
        deriv_syn0_tmp_cof = hybolic_alpha * hybolic_alpha / 4;
        for (c = 0; c < layer1_size; c++) deriv_syn0_poincare[c] = deriv_syn0_tmp_cof * deriv_syn0_poincare[c];
        for (c = 0; c < layer1_size; c++) neu1e[c] += deriv_syn0_poincare[c];
        deriv_syn0_tmp_cof = 0;
        deriv_syn1neg_tmp_cof = 0;

        for (c = 0; c < layer1_size; c++) deriv_syn1neg_tmp_cof += syn0[c + l1] * syn1neg[c + l2];
        deriv_syn1neg_tmp_cof = (hybolic_syn0_norm - 2 * deriv_syn1neg_tmp_cof + 1) / hybolic_beta /hybolic_beta;
        for (c = 0; c < layer1_size; c++) deriv_syn1neg_tmp_arr[c] = syn1neg[c + l2] * deriv_syn1neg_tmp_cof;
        for (c = 0; c < layer1_size; c++) deriv_syn0_tmp_arr[c] = syn0[c + l1] / hybolic_beta;
    	deriv_syn1neg_tmp_cof = g * 4 / hybolic_alpha / sqrt(hybolic_gamma * hybolic_gamma - 1);
        for (c = 0; c < layer1_size; c++) deriv_syn1neg_poincare[c] = deriv_syn1neg_tmp_cof * (deriv_syn0_tmp_arr[c] - deriv_syn1neg_tmp_arr[c]);
        if (deriv_syn1neg_poincare[0] == 0){
          for (c = 0; c < layer1_size; c++) if (deriv_syn1neg_poincare[c] == 0) deriv_syn1neg_poincare[c] = 0.00001;
        } 	
        deriv_syn1neg_tmp_cof = hybolic_beta * hybolic_beta / 4;
        for (c = 0; c < layer1_size; c++) deriv_syn1neg_poincare[c] = deriv_syn1neg_tmp_cof * deriv_syn1neg_poincare[c];
        for (c = 0; c < layer1_size; c++) map_tmp_arr[c] = syn1neg[c + l2];
        map_arr = ExponentialMap(map_tmp_arr, deriv_syn1neg_poincare);
        for (c = 0; c < layer1_size; c++) syn1neg[c + l2] = map_arr[c];
        free(map_arr);
      }

      for (c = 0; c < layer1_size; c++) map_tmp_arr[c] = syn0[c + l1];
      map_arr = ExponentialMap(map_tmp_arr, neu1e);
      for (c = 0; c < layer1_size; c++) syn0[c + l1] = map_arr[c];
      free(map_arr);
    }

    sentence_position++;
    if (sentence_position >= sentence_length) {
      sentence_length = 0;
      continue;
    }
  }
  fclose(fi);
  free(neu1);
  free(neu1e);
  free(deriv_syn0_tmp_arr);
  free(deriv_syn1neg_tmp_arr);
  free(deriv_syn0_poincare);
  free(deriv_syn1neg_poincare);
  pthread_exit(NULL);
}

void TrainModel() {
  long a, b;
  FILE *fp;
  pthread_t *pt = (pthread_t *)malloc(num_threads * sizeof(pthread_t));
  printf("Starting training using file %s\n", train_file);
  starting_alpha = alpha;
  if (read_vocab_file[0] != 0) ReadVocab(); else LearnVocabFromTrainFile();
  if (save_vocab_file[0] != 0) SaveVocab(); 
  if (output_file[0] == 0) return;
  InitNet();
  if (negative > 0) InitUnigramTable();
  start = clock();
  for (a = 0; a < num_threads; a++) pthread_create(&pt[a], NULL, TrainModelThread, (void *)a);
  for (a = 0; a < num_threads; a++) pthread_join(pt[a], NULL);
  char txt[5];
  strcpy(txt, ".txt\0");
  fp = fopen(strcat(output_file, txt), "wb");
  fprintf(fp, "%lld %lld\n", vocab_size, layer1_size);
  for (a = 0; a < vocab_size; a++) {
    fprintf(fp, "%s ", vocab[a].word);
    for (b = 0; b < layer1_size; b++) fprintf(fp, "%lf ", syn0[a * layer1_size + b]);
    fprintf(fp, "\n");
  }
  fclose(fp);
}

int ArgPos(char *str, int argc, char **argv) {
  int a;
  for (a = 1; a < argc; a++) if (!strcmp(str, argv[a])) {
    if (a == argc - 1) {
      printf("Argument missing for %s\n", str);
      exit(1);
    }
    return a;
  }
  return -1;
}

int main(int argc, char **argv) {
  int i;
  if (argc == 1) {
    printf("---HHNE by Yiding Zhang---\n");
    printf("---The code and following instructions are built upon word2vec.c by Mikolov et al.---\n\n");

    printf("Options:\n");
    printf("Parameters for training:\n");
    printf("\t-train <file>\n");
    printf("\t\tUse text data from <file> to train the model\n");
    printf("\t-output <file>\n");
    printf("\t\tUse <file> to save the resulting word vectors / word clusters\n");
    printf("\t-size <int>\n");
    printf("\t\tSet size of word vectors; default is 100\n");
    printf("\t-window <int>\n");
    printf("\t\tSet max skip length between words; default is 5\n");
    printf("\t-sample <float>\n");
    printf("\t\tSet threshold for occurrence of words. Those that appear with higher frequency in the training data\n");
    printf("\t\twill be randomly down-sampled; default is 1e-3, useful range is (0, 1e-5)\n"); 
    printf("\t-negative <int>\n");
    printf("\t\tNumber of negative examples; default is 5, common values are 3 - 10 (0 = not used)\n");
    printf("\t-threads <int>\n");
    printf("\t\tUse <int> threads (default 12)\n");
    printf("\t-iter <int>\n");
    printf("\t\tRun more training iterations (default 5)\n");
    printf("\t-min-count <int>\n");
    printf("\t\tThis will discard words that appear less than <int> times; default is 5\n");
    printf("\t-alpha <float>\n");
    printf("\t\tSet the starting learning rate; default is 0.025 for skip-gram\n");
    printf("\t-debug <int>\n");
    printf("\t\tSet the debug mode (default = 2 = more info during training)\n");
    printf("\t-save-vocab <file>\n");
    printf("\t\tThe vocabulary will be saved to <file>\n");
    printf("\t-read-vocab <file>\n");
    printf("\t\tThe vocabulary will be read from <file>, not constructed from the training data\n");
    printf("\nExamples:\n");
    printf("./hhne -train random_walks.txt -output hhne.embeddings -size 2 -window 5 -negative 10 -threads 32 -iter 5 -alpha 0.025\n\n");
    return 0;
  }
  output_file[0] = 0;
  save_vocab_file[0] = 0;
  read_vocab_file[0] = 0;
  if ((i = ArgPos((char *)"-size", argc, argv)) > 0) layer1_size = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-train", argc, argv)) > 0) strcpy(train_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-save-vocab", argc, argv)) > 0) strcpy(save_vocab_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-read-vocab", argc, argv)) > 0) strcpy(read_vocab_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-debug", argc, argv)) > 0) debug_mode = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-alpha", argc, argv)) > 0) alpha = atof(argv[i + 1]);
  if ((i = ArgPos((char *)"-output", argc, argv)) > 0) strcpy(output_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-window", argc, argv)) > 0) window = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-sample", argc, argv)) > 0) sample = atof(argv[i + 1]);
  if ((i = ArgPos((char *)"-negative", argc, argv)) > 0) negative = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-threads", argc, argv)) > 0) num_threads = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-iter", argc, argv)) > 0) iter = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-min-count", argc, argv)) > 0) min_count = atoi(argv[i + 1]);
  vocab = (struct vocab_word *)calloc(vocab_max_size, sizeof(struct vocab_word));
  vocab_hash = (int *)calloc(vocab_hash_size, sizeof(int));
  expTable = (real *)malloc((EXP_TABLE_SIZE + 1) * sizeof(real));
  for (i = 0; i < EXP_TABLE_SIZE; i++) {
    expTable[i] = exp((i / (real)EXP_TABLE_SIZE * 2 - 1) * MAX_EXP); // Precompute the exp() table
    expTable[i] = expTable[i] / (expTable[i] + 1);                   // Precompute f(x) = x / (x + 1)
  }
  TrainModel();
  return 0;
}
