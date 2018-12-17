###################################
# CS B551 Fall 2018, Assignment #3
#
# Your names and user ids: Aarushi Bisht(abisht) Jagpreet Singh Chawla(jchawla)
#
# (Based on skeleton code by D. Crandall)
#
#
####
# For Training following probabilities are calculated
# 1. Emission Probability(P(W=w|S=T) = Count of occurrence of word = W with tag = T/ Count of occurrences of tag = T

# 2. Transition Probability according to bayes net in figure 1a
# P(Sn+1 = T1|Sn = T2 = Count of occurrences of T2 followed by T1/ Count of occurrences of T2

# 3. Transition Probability according to bayes net in figure 1c
# P(Sn = T1|Sn-1 = T2, Sn-2=T3)= Count of occurrences of T1 followed by T2, T3/Count of occurrences of T2 followed by T3

# 4. Probability of occurrence of each tag
# P(S=t) = Count of occurrences of tag t/ Total words

# For the missing emission, transition probabilities laplace smoothing is used.
# Laplace emission prob
# P(W= w|S = T) = (Count of word given a tag + 0.001)/(Count of tag + 0.001 * (number of unique words + 1))
# Laplace transition prob
# P(Sn = tag1| Sn-1 = tag2) = (Count of occurrences of tag1 followed by tag2 + 0.001)/(Count of tag2 + 0.001 * number of unique tags + 1))

# SIMPLE MODEL
# P(S=t| W=W1) = P(W=W1|S=t) * P(S=t). For each word in the sentence, this value is calculated for all possible tags and
# the one with maximum value is selected.

# HMM
# For each word in the sentence, probability for each tag is calculated recursively as
# vn = P(Wn|Sn) * max of {P(Sn|Sn-1) * vn-1}
# At the end of the sentence highest probability is selected from the vn array. vn-1 used for calculating the selected
# vn is then selected for wn-1 tag. This is done recursively ie vn-2 used for calculating the selected vn-1 is chosen
# for word = wn-2 so on.

# MCMC
# A random sequence of tags is generated.
# Each tag in the sequence is now sampled according to the prior probability of
# P(Sn=t|Wn=W, Sn-1=t1, Sn-2=t2) = P(Wn=W|Sn=t) * P(Sn=t|Sn-1=t1,Sn-2=t2) * P(Sn-1=t1|Sn-2=t2) * P(Sn-2=t2) / P(Sn-1=t1, Sn-2=t2, Wn = W)
# A warm up period of 2000 samples is set and in-total 8000 samples are generated
# From the sampled set, tag with most probability is selected for each word of the sentence

# To simplify the model sentence tag sequence is appended with a start tag and end tag.
####

import random
import math


# We've set up a suggested code structure, but feel free to change it. Just
# make sure your code still works with the label.py and pos_scorer.py code
# that we've supplied.
#
import random
import math

class Solver:
    tags_prob = {}
    tags_prob["__start1__"] = 1
    tags_prob["__start2__"] = 1
    tags_prob["__end1__"] = 1
    tags_prob["__end2__"] = 1
    transition_prob = {}
    emission_prob = {}
    total_words = 0
    complex_emission_prob = {}
    list_of_tags = []
    total_unique_words = 0
    total_sample = []

    # Calculate the log of the posterior probability of a given sentence
    #  with a given part-of-speech labeling. Right now just returns -999 -- fix this!
    def posterior(self, model, sentence, label):
        if model == "Simple":
            return self.simple_log_probability(sentence, label)
        elif model == "Complex":
            return self.complex_log_probability(sentence, label)
        elif model == "HMM":
            return self.hmm_log_probability(sentence, label)
        else:
            print("Unknown algo!")

    # calculates the sum of posterior probability of a label given a word
    def simple_log_probability(self, sentence, label):
        sum = 0
        for i in range(0, len(sentence)):
            sum += self.get_simple_prob_given_label(sentence[i], label[i])
        return sum

    # calculates the log of posterior probability using simple model
    def get_simple_prob_given_label(self, word, label):
        return math.log(self.get_laplace_smoothened_emission_prob(word, label)) + math.log(self.tags_prob[label])

    # probability of a label given a tag is calculated as maximum of P(W=W1|S=t) * P(S=t) for all possible tags
    def simplified(self, sentence):
        tags = []
        for i in range(0, len(sentence)):
            highest_prob = -99999
            probable_tag = None
            for tag in self.list_of_tags:
                prob = self.get_simple_prob_given_label(sentence[i], tag)
                if prob > highest_prob:
                    highest_prob = prob
                    probable_tag = tag
            tags.append(probable_tag)
        return tags

    # Runs the viterbi algorithm to return the most probable sequence
    def hmm_viterbi(self, sentence):
        v = []
        v1 = []
        for tag in self.list_of_tags:
            v1.append(math.log(self.tags_prob[tag]) + math.log(self.get_laplace_smoothened_emission_prob(sentence[0], tag)))
        v.append(v1)
        v, selected_tags = self.run_viterbi(sentence, v, [])
        return self.get_most_probable_sequence(v, selected_tags)

    # For each word in the sentence, probability for each tag is calculated recursively as
    # vn = P(Wn|Sn) * max of {P(Sn|Sn-1) * vn-1}
    def run_viterbi(self, sentence, v, selected_tags):
        if len(v) == len(sentence):
            return v, selected_tags
        t = []
        v_present = []
        iteration_n = len(v)
        for tag in self.list_of_tags:
            selected_tag, max_v = self.max_of_previous_tag(tag, v[-1])
            t.append(selected_tag)
            v_present.append(math.log(self.get_laplace_smoothened_emission_prob(sentence[iteration_n], tag)) + max_v)
        selected_tags.append(t)
        v.append(v_present)
        return self.run_viterbi(sentence, v, selected_tags)

    # returns the maximum of P(Wn|Wn-1) * vn-1 and tag index giving the max value
    def max_of_previous_tag(self, tag, v_previous):
        max_v = -99999
        max_tag = None
        for i in range(0, len(v_previous)):
            p = math.log(self.get_laplace_smoothened_transition_prob(tag, self.list_of_tags[i])) + v_previous[i]
            if p > max_v:
                max_v = p
                max_tag = i
        return max_tag, max_v

    # selects the tag with maximum prob for last word and iterates the selected_tags list to get rest of the tags
    # returns most probable sequence for a the sentence
    def get_most_probable_sequence(self, v, selected_tags):
        most_probable_sequence = []
        max_j = -1
        for j in range(0, len(v[-1])):
            if v[-1][max_j] < v[-1][j]:
                max_j = j
        most_probable_sequence.append(self.list_of_tags[max_j])
        for i in range(len(selected_tags)-1, -1, -1):
            max_j = selected_tags[i][max_j]
            most_probable_sequence.append(self.list_of_tags[max_j])
        most_probable_sequence.reverse()
        return most_probable_sequence

    # returns the sum of posterior probabilities calculated as follows P(S=t|w=w3) = log(P(w=w3|S=t) + log(P(w=w3|w=w2)
    def hmm_log_probability(self, sentence, label):
        sum = math.log(self.tags_prob[label[0]]) + math.log(self.get_laplace_smoothened_emission_prob(sentence[0], label[0]))
        for i in range(0, len(sentence)):
            sum += math.log(self.get_laplace_smoothened_emission_prob(sentence[i], label[i])) + math.log(self.get_laplace_smoothened_transition_prob(label[i], label[i-1]))
        return sum

    # returns the best sequence calculated using MCMC
    # Added "__start1__", "__start2__", "__end1__", "__end2__" before and send send of each label sequence generated to
    # determine tags most probable in the start or tags most probable in the end

    def complex_mcmc(self, sentence):
        initial_seq = ["__start2__", "__start1__"]
        for i in range(0, len(sentence)):
            initial_seq.append(self.list_of_tags[random.randint(0, len(self.list_of_tags) - 1)])
        initial_seq.append("__end1__")
        initial_seq.append("__end2__")
        self.total_sample = self.sample(initial_seq, sentence)
        return self.get_best_labeling(sentence)

    # return samples
    def sample(self, initial_sample, sentence):
        total_samples = []
        warm_up_period = 1000
        current_sample = list(initial_sample)
        for i in range(0, 5000):
            for sampling_index in range(2, len(current_sample) - 2):
                # get the prior probabilities required for sampling
                prior_probs = self.get_prior_prob(current_sample, sampling_index, sentence)
                # sample the tag
                sampled_tag = self.sample_tag(prior_probs)
                # update the sample
                current_sample[sampling_index] = sampled_tag
            if i > warm_up_period:
                total_samples.append(current_sample)
            current_sample = list(current_sample)
        return total_samples

    # returns the probability needed for the sampling
    # For each tag probability is calculated as follows:
    # P(Sn=t|Wn=W, Sn-1=t1, Sn-2=t2) = P(Wn=W|Sn=t) * P(Sn=t|Sn-1=t1,Sn-2=t2) * P(Sn-1=t1|Sn-2=t2) * P(Sn-2=t2) / P(Sn-1=t1, Sn-2=t2, Wn = W)
    def get_prior_prob(self, initial_sample, sampling_index, sentence):
        prior_probs = []
        for tag in self.list_of_tags:
            prior_probs.append(self.get_numerator(tag, initial_sample[sampling_index-1], initial_sample[sampling_index-2], sentence[sampling_index-2]))
        denominator = sum(prior_probs)
        return [n/denominator for n in prior_probs]

    def get_numerator(self, tag, known_tag1, known_tag2, word):
        num = self.get_laplace_smoothened_emission_prob(word, tag) * self.get_laplace_smoothened_complex_prb(tag, known_tag1, known_tag2) * self.get_laplace_smoothened_transition_prob(known_tag1, known_tag2) * self.tags_prob[known_tag2]
        return num

    # select a tag
    def sample_tag(self, prior_probs):
        prior_probs_sorted = sorted(prior_probs)
        cumulative_probs = [prior_probs_sorted[0]]
        for i in range(1, len(prior_probs_sorted)):
            cumulative_probs.append(cumulative_probs[-1] + prior_probs_sorted[i])
        random_prob = random.random()
        for j in range(0, len(cumulative_probs)):
            if cumulative_probs[j] > random_prob:
                break
        index = prior_probs.index(prior_probs_sorted[j])
        return self.list_of_tags[index]

    # calculates the probability of a tag for a word by counting the occurrences of the tag in the sample
    def get_tag_prob(self, tag, j):
        count = 0
        for i in range(0, len(self.total_sample)):
            if self.total_sample[i][j] == tag:
                count += 1
        if count == 0:
            return 1e-20
        return count/len(self.total_sample)

    # select the tag with maximum probability
    def get_best_labeling(self, sentence):
        result = []
        for j in range(0, len(sentence)):
            max_prob = 0
            selected_tag = None
            for tag in self.list_of_tags:
                prob = self.get_tag_prob(tag, j+2)
                if prob > max_prob:
                    max_prob = prob
                    selected_tag = tag
            result.append(selected_tag)
        return result

    def complex_log_probability(self, sentence, label):
        sum = 0
        for i in range(0, len(sentence)):
            sum += math.log(self.get_tag_prob(label[i], i+2))
        return sum

    # count the total number of words
    def _populate_total_words(self, data):
        seen = []
        for i in range(0, len(data)):
            self.total_words += len(data[i][0])
            for j in range(0, len(data[i][0])):
                if data[i][0][j] in seen:
                    continue
                else:
                    self.total_unique_words += 1

    #  to find the probability of each tag and store list of unique tags
    def _populate_prob_of_tags_and_tag_list(self, data):
        for i in range(0, len(data)):
            for j in range(0, len(data[i][1])):
                if data[i][1][j] in self.tags_prob:
                    self.tags_prob[data[i][1][j]] += 1/self.total_words
                else:
                    self.list_of_tags.append(data[i][1][j])
                    self.tags_prob[data[i][1][j]] = 1/self.total_words

    # to calculate transitional probability
    def _populate_transition_prob(self, data):
        for i in range(0, len(data)):
            for j in range(0, len(data[i][1])):
                if j == 0:
                    if data[i][1][j] + '-' + "__start1__" in self.transition_prob:
                        self.transition_prob[data[i][1][j] + '-' + "__start1__"] += 1 / len(data)
                    else:
                        self.transition_prob[data[i][1][j] + '-' + "__start1__"] = 1 / len(data)
                elif j == len(data[i][1]) - 1:
                    if "__end1__" + '-' + data[i][1][j] in self.transition_prob:
                        self.transition_prob["__end1__" + '-' + data[i][1][j]] += 1 / (self.tags_prob[data[i][1][j]] * self.total_words)
                    else:
                        self.transition_prob["__end1__" + '-' + data[i][1][j]] = 1 / (self.tags_prob[data[i][1][j]] * self.total_words)
                else:
                    if data[i][1][j] + '-' + data[i][1][j-1] in self.transition_prob:
                        self.transition_prob[data[i][1][j] + '-' + data[i][1][j-1]] += 1/(self.tags_prob[data[i][1][j-1]] * self.total_words)
                    else:
                        self.transition_prob[data[i][1][j] + '-' + data[i][1][j-1]] = 1/(self.tags_prob[data[i][1][j-1]] * self.total_words)

    # gives emission probabilities
    def _populate_emission_prob(self, data):
        for i in range(0, len(data)):
            for j in range(0, len(data[i][0])):
                if data[i][0][j] + '-' + data[i][1][j] in self.emission_prob:
                    self.emission_prob[data[i][0][j] + '-' + data[i][1][j]] += 1/(self.tags_prob[data[i][1][j]] * self.total_words)
                else:
                    self.emission_prob[data[i][0][j] + '-' + data[i][1][j]] = 1/(self.tags_prob[data[i][1][j]] * self.total_words)

    # to calculate emission probabilities according the the complex model
    def _populate_complex_transition_prob(self, data):
        for i in range(0, len(data)):
            for j in range(0, len(data[i][1])):
                if j == 0:
                    if data[i][1][j] + '-' + "__start1__" + "-" + "__start2__" in self.complex_emission_prob:
                        self.complex_emission_prob[data[i][1][j] + '-' + "__start1__" + "-" + "__start2__"] += 1/len(data)
                    else:
                        self.complex_emission_prob[data[i][1][j] + '-' + "__start1__" + "-" + "__start2__"] = 1/len(data)
                elif j == 1:
                    if data[i][1][j] + '-' + data[i][1][j-1] + "-" + "__start1__" in self.complex_emission_prob:
                        self.complex_emission_prob[data[i][1][j] + '-' + data[i][1][j-1] + "-" + "__start1__"] += 1/(self.transition_prob[data[i][1][j-1] + "-" + "__start1__"]*len(data))
                    else:
                        self.complex_emission_prob[data[i][1][j] + '-' + data[i][1][j-1] + "-" + "__start1__"] = 1/(self.transition_prob[data[i][1][j-1] + "-" + "__start1__"]*len(data))
                else:
                    if data[i][1][j] + '-' + data[i][1][j-1] + "-" + data[i][1][j-2] in self.complex_emission_prob:
                        self.complex_emission_prob[data[i][1][j] + '-' + data[i][1][j-1] + "-" + data[i][1][j-2]] += 1/(self.transition_prob[data[i][1][j-1] + "-" + data[i][1][j-2]] * self.tags_prob[data[i][1][j-2]] * self.total_words)
                    else:
                        self.complex_emission_prob[data[i][1][j] + '-' + data[i][1][j-1] + "-" + data[i][1][j-2]] = 1/(self.transition_prob[data[i][1][j-1] + "-" + data[i][1][j-2]] * self.tags_prob[data[i][1][j-2]] * self.total_words)

            if len(data[i][1]) > 1:
                if "__end1__" + '-' + data[i][1][-1] + '-' + data[i][1][-2] in self.complex_emission_prob:
                    self.complex_emission_prob["__end1__" + '-' + data[i][1][-1] + '-' + data[i][1][-2]] += 1/(self.transition_prob[data[i][1][-1] + '-' + data[i][1][-2]] * self.tags_prob[data[i][1][-2]] * self.total_words)
                else:
                    self.complex_emission_prob["__end1__" + '-' + data[i][1][-1] + '-' + data[i][1][-2]] = 1/(self.transition_prob[data[i][1][-1] + '-' + data[i][1][-2]] * self.tags_prob[data[i][1][-2]] * self.total_words)

            if "__end2__" + '-' + "__end1__" + '-' + data[i][1][-1] in self.complex_emission_prob:
                self.complex_emission_prob["__end2__" + '-' + "__end1__" + '-' + data[i][1][-1]] += 1/(self.transition_prob["__end1__" + '-' + data[i][1][-1]] * self.tags_prob[data[i][1][-1]] * self.total_words)
            else:
                self.complex_emission_prob["__end2__" + '-' + "__end1__" + '-' + data[i][1][-1]] = 1/(self.transition_prob["__end1__" + '-' + data[i][1][-1]] * self.tags_prob[data[i][1][-1]] * self.total_words)

    # Do the training!
    #
    def train(self, data):
        self._populate_total_words(data)
        self._populate_prob_of_tags_and_tag_list(data)
        self._populate_transition_prob(data)
        self._populate_emission_prob(data)
        self._populate_complex_transition_prob(data)

    def get_laplace_smoothened_emission_prob(self, word, tag):
        if word + '-' + tag in self.emission_prob:
            tmp = self.emission_prob[word + '-' + tag] * self.tags_prob[tag] * self.total_words
        else:
            tmp = 0
        return (tmp + 0.001)/((self.tags_prob[tag] * self.total_words) + 0.001 * (self.total_unique_words + 1))


    def get_laplace_smoothened_transition_prob(self, tag1, tag2):
        if tag1 + '-' + tag2 in self.transition_prob:
            tmp = self.transition_prob[tag1 + '-' + tag2] * self.tags_prob[tag2] * self.total_words
        else:
            tmp = 0
        return (tmp + 0.001)/((self.tags_prob[tag2] * self.total_words) + 0.001 * (len(self.list_of_tags) + 1))

    def get_laplace_smoothened_complex_prb(self, tag, known_tag1, known_tag2):
        tmp = 0
        alpha = 0.001
        if tag + '-' + known_tag1 + '-' + known_tag2 in self.complex_emission_prob:
            tmp = self.complex_emission_prob[tag + '-' + known_tag1 + '-' + known_tag2] * self.get_laplace_smoothened_transition_prob(known_tag1, known_tag2) * self.tags_prob[known_tag2] * self.total_words
        return (tmp + alpha)/((self.get_laplace_smoothened_transition_prob(known_tag1, known_tag2) * self.tags_prob[known_tag2] * self.total_words) + (alpha * len(self.list_of_tags) + 1))


    # This solve() method is called by label.py, so you should keep the interface the
    #  same, but you can change the code itself. 
    # It should return a list of part-of-speech labelings of the sentence, one
    #  part of speech per word.
    #
    def solve(self, model, sentence):
        if model == "Simple":
            return self.simplified(sentence)
        elif model == "Complex":
            return self.complex_mcmc(sentence)
        elif model == "HMM":
            return self.hmm_viterbi(sentence)
        else:
            print("Unknown algo!")

