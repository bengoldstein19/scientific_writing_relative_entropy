from bs4 import BeautifulSoup
import csv
import os
import matplotlib.pyplot as plt
import json
import scipy.stats as stats
import math

# MARK: REQUIRED FILE PATHS
ROOT_DIR = "/Users/benjamingoldstein/Downloads"
METADATA_FILENAME = "Royal_Society_Corpus_open_v6.0_meta.tsv" 
TEXT_DIR = "Royal_Society_Corpus_open_v6.0_texts_tei" #relative to root_dir
TEXT_BASE_FILENAME = "Royal_Society_Corpus_open_v6.0_text_{}.tei.xml" #text ID inserted into brackets

# MARK: OUTPUT FILENAMES
TOPIC_DATA_OUTPUT_FILENAME = "topic_prevalence_data.csv"
RELATIVE_ENTROPY_OUTPUT_FILENAME = "relative_entropies_by_topic_vs_time.csv"
PLOT_BASE_FILENAME = "Entropy_vs_Time_{}"

# MARK: COLUMN NAMES IN CORPUS
ID_COL = "id"
PRIMARY_TOPIC_COL = "primaryTopic"
SECONDARY_TOPIC_COL = "secondaryTopic"
TYPE_COL = "type"
DECADE_COL = "decade"

# MARK: CORPUS CONSTANTS
START_DECADE = 1660
END_DECADE = 1920
DECADES = list(range(START_DECADE, END_DECADE + 1, 10))
NUM_DECADES = int((END_DECADE - START_DECADE)/10) + 1

#PARAMETER CONSTANTS
COMMON_WORDS = ["a", "able", "about", "across", "after", "all", "almost", "also", "am", "among", "an", "and" , "any", "are", "as", "at", "be", "because", "been", "but", "by", "can", "cannot", "could", "dear", "did", "do", "does", "either", "else", "ever", "every", "for", "from", "get", "got", "had", "has", "have", "he", "her", "hers", "him", "his", "how", "however", "i", "if", "in" , "into", "is" , "it", "its", "just", "least", "let", "like", "likely", "may", "me", "might", "most", "must", "my", "neither", "no", "nor", "not", "of", "off", "often", "on", "only", "or" , "other", "our", "own", "rather", "said", "say", "says", "she", "should", "since", "so", "some", "than", "that", "the", "their", "them", "then", "there", "these", "they", "this", "tis", "to", "too", "twas", "us", "wants", "was", "we", "were", "what", "when", "where", "which", "while", "who", "whom", "why", "will", "with", "would", "yet", "you", "your"]
# https://www.textfixer.com/tutorials/common-english-words.txt (common words url)
MIN_LEMMA_LEN = 3
INITIAL_FREQUENCY_DIFF_FILTER = 2000 # (occurrences/million)
GOAL_P_VAL = 0.01
MIN_ACCEPTED_P_VAL = 0.05

# MARK: PLOTTING CONSTANTS
PLOT_COLORS = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
PLOT_TEXTURES = ['-', '--', '-.', ':', 'o']
NUM_COLORS = len(PLOT_COLORS)
NUM_TEXTURE = len(PLOT_TEXTURES)
PRIMARY_PLOT_COL = 0
SECONDARY_PLOT_COL = 1

#MARK: DATA STRUCTURE CONSTANTS
PRIMARY_TOPIC_IND = 0
SECONDARY_TOPIC_IND = 1
OUTPUT_FILE_TOPIC_COL = 0


def perform_topic_analysis():
	metadata_filepath = os.path.join(ROOT_DIR, METADATA_FILENAME)

	with open(metadata_filepath, "r") as metadata_filepointer:
		# create reader object
		metadata_reader = csv.reader(metadata_filepointer, delimiter='\t')

		# initialize structures for iteration
		header_row = True
		index_table = {}  # map of header to its index in the row
		# map of topic to list of its frequencies in each decade stored as array of tuples (primary, secondary)
		topic_data = {}
		total_primary_topics = [0] * NUM_DECADES
		total_secondary_topics = [0] * NUM_DECADES
		total_primary_topics_unique = [0] * NUM_DECADES
		total_secondary_topics_unique = [0] * NUM_DECADES

		# iterate over reader object
		for row in metadata_reader:
			# Load headers and their indices into structure (storing indices allows flexibility/performance with large # of columns)
			if header_row:
				ind = 0
				for header in row:
					index_table[header] = int(ind)
					ind += 1
				header_row = False

			elif row[index_table[TYPE_COL]] == "fla" or row[index_table[TYPE_COL]] == "article":
				# get relevant info from row
				primary_topic = row[index_table[PRIMARY_TOPIC_COL]]
				secondary_topic = row[index_table[SECONDARY_TOPIC_COL]]
				decade = int(row[index_table[DECADE_COL]])
				decade_index = int((decade - START_DECADE)/10)

				# initialize arrays in topic data if not already there
				if primary_topic and not topic_data.get(primary_topic):
					topic_data[primary_topic] = [
						[0, 0] for _ in range(NUM_DECADES)]

				if secondary_topic and not topic_data.get(secondary_topic):
					topic_data[secondary_topic] = [
						[0, 0] for _ in range(NUM_DECADES)]

				# increment topic data values for primary topic and secondary topic for appropriate decade
				if primary_topic:
					if (topic_data[primary_topic][decade_index][PRIMARY_TOPIC_IND] == 0):
						total_primary_topics_unique[decade_index] += 1
					topic_data[primary_topic][decade_index][PRIMARY_TOPIC_IND] += 1
					total_primary_topics[decade_index] += 1
				if secondary_topic:
					if (topic_data[secondary_topic][decade_index][SECONDARY_TOPIC_IND] == 0):
						total_secondary_topics_unique[decade_index] += 1
					topic_data[secondary_topic][decade_index][SECONDARY_TOPIC_IND] += 1
					total_secondary_topics[decade_index] += 1

		fig, ax = plt.subplots(nrows=1, ncols=2)

		for i, k in enumerate(topic_data.keys()):
			ax[PRIMARY_PLOT_COL].plot(DECADES, [x[PRIMARY_TOPIC_IND] for x in topic_data[k]],
									  f"{PLOT_COLORS[i % NUM_COLORS]}{PLOT_TEXTURES[int(i / NUM_COLORS)]}", label=k)
			ax[SECONDARY_PLOT_COL].plot(DECADES, [x[SECONDARY_TOPIC_IND] for x in topic_data[k]],
										f"{PLOT_COLORS[i % NUM_COLORS]}{PLOT_TEXTURES[int(i / NUM_COLORS)]}", label=k)

		ax[PRIMARY_PLOT_COL].set_title(
			"Occurrences of Primary Topic vs Time")
		ax[PRIMARY_PLOT_COL].set_ylabel("# Occurrences")
		ax[PRIMARY_PLOT_COL].set_xlabel("Decade")

		ax[SECONDARY_PLOT_COL].set_title(
			"Occurrences of Secondary Topic vs Time")
		ax[SECONDARY_PLOT_COL].set_ylabel("# Occurrences")
		ax[SECONDARY_PLOT_COL].set_xlabel("Decade")

		handles, labels = ax[PRIMARY_PLOT_COL].get_legend_handles_labels()

		plt.tight_layout()

		plt.legend(handles, labels, loc='upper center', bbox_to_anchor=(-0.25, 1),
				   fancybox=True, shadow=False, ncol=6)

		fig, ax = plt.subplots(nrows=1, ncols=2)

		for i, k in enumerate(topic_data.keys()):
			ax[PRIMARY_PLOT_COL].plot(DECADES, [x[PRIMARY_TOPIC_IND]/total_primary_topics[i] for i, x in enumerate(topic_data[k])],
									  f"{PLOT_COLORS[i % NUM_COLORS]}{PLOT_TEXTURES[int(i / NUM_COLORS)]}", label=k)
			ax[SECONDARY_PLOT_COL].plot(DECADES, [x[SECONDARY_TOPIC_IND]/total_secondary_topics[i] for i, x in enumerate(topic_data[k])],
										f"{PLOT_COLORS[i % NUM_COLORS]}{PLOT_TEXTURES[int(i / NUM_COLORS)]}", label=k)

		ax[PRIMARY_PLOT_COL].set_title(
			"Share of Primary Topic vs Time")
		ax[PRIMARY_PLOT_COL].set_ylabel("% Share")
		ax[PRIMARY_PLOT_COL].set_xlabel("Decade")

		ax[SECONDARY_PLOT_COL].set_title(
			"Share of Secondary Topic vs Time")
		ax[SECONDARY_PLOT_COL].set_ylabel("% Share")
		ax[SECONDARY_PLOT_COL].set_xlabel("Decade")

		plt.tight_layout()

		plt.legend(handles, labels, loc='upper center', bbox_to_anchor=(-0.25, 1),
				   fancybox=True, shadow=False, ncol=6)

		# plt.figure(3)

		fig, ax = plt.subplots(nrows=1, ncols=2)

		ax[PRIMARY_PLOT_COL].plot(DECADES, total_primary_topics_unique, 'r-')
		ax[SECONDARY_PLOT_COL].plot(
			DECADES, total_secondary_topics_unique, 'b-')
		ax[PRIMARY_PLOT_COL].set_title(
			"Number of Primary Topics vs Time")
		ax[PRIMARY_PLOT_COL].set_ylabel("# Topics")
		ax[PRIMARY_PLOT_COL].set_xlabel("Decade")

		ax[SECONDARY_PLOT_COL].set_title(
			"Number of Secondary Topics vs Time")
		ax[SECONDARY_PLOT_COL].set_ylabel("# Topics")
		ax[SECONDARY_PLOT_COL].set_xlabel("Decade")

		plt.tight_layout()

		fig, ax = plt.subplots(nrows=1, ncols=2)

		ax[PRIMARY_PLOT_COL].plot(DECADES, [
								  total_primary_topics_unique[i]/total_primary_topics[i] for i in range(NUM_DECADES)], 'r-')
		ax[SECONDARY_PLOT_COL].plot(DECADES, [
									total_secondary_topics_unique[i]/total_secondary_topics[i] for i in range(NUM_DECADES)], 'b-')
		ax[PRIMARY_PLOT_COL].set_title(
			"# Primary Topics : Articles Ratio vs Time")
		ax[PRIMARY_PLOT_COL].set_ylabel("Topics/Article")
		ax[PRIMARY_PLOT_COL].set_xlabel("Decade")

		ax[SECONDARY_PLOT_COL].set_title(
			"# Secondary Topics : Articles Ratio vs Time")
		ax[SECONDARY_PLOT_COL].set_ylabel("Topics/Article")
		ax[SECONDARY_PLOT_COL].set_xlabel("Decade")

		plt.tight_layout()
		plt.show()

		with open(os.path.join(ROOT_DIR, TOPIC_DATA_OUTPUT_FILENAME), "w") as topic_fp:
			topic_writer = csv.writer(topic_fp)
			topic_writer.writerow(["Year", *DECADES])
			topic_writer.writerow(['-----'] * NUM_DECADES)
			topic_writer.writerow(["OCCURRENCES AS PRIMARY TOPIC"])
			for k in topic_data.keys():
				topic_writer.writerow([k] + [x[PRIMARY_TOPIC_IND] for x in topic_data[k]])
			topic_writer.writerow(['-----'] * NUM_DECADES)
			topic_writer.writerow(["OCCURRENCES AS SECONDARY TOPIC"])
			for k in topic_data.keys():
				topic_writer.writerow([k] + [x[SECONDARY_TOPIC_IND] for x in topic_data[k]])
			topic_writer.writerow(['-----'] * NUM_DECADES)
			topic_writer.writerow(["SHARE OF PRIMARY TOPIC REFERENCES"])
			for k in topic_data.keys():
				topic_writer.writerow(
					[k] + [x[0]/total_primary_topics[i] for i, x in enumerate(topic_data[k])])
			topic_writer.writerow(['-----'] * NUM_DECADES)
			topic_writer.writerow(["SHARE OF SECONDARY TOPIC REFERENCES"])
			for k in topic_data.keys():
				topic_writer.writerow(
					[k] + [x[1]/total_secondary_topics[i] for i, x in enumerate(topic_data[k])])
			topic_writer.writerow(['-----'] * NUM_DECADES)
			topic_writer.writerow(['-----'] * NUM_DECADES)
			topic_writer.writerow(
				["Total Primary Topics Appearing"] + total_primary_topics_unique)
			topic_writer.writerow(
				["Total Secondary Topics Appearing", *total_secondary_topics_unique])
			topic_writer.writerow(["Primary Topics/Article Ratio"] + [
								  total_primary_topics_unique[i]/total_primary_topics[i] for i in range(NUM_DECADES)])
			topic_writer.writerow(["Secondary Topics/Article Ratio"] + [
								  total_secondary_topics_unique[i]/total_secondary_topics[i] for i in range(NUM_DECADES)])


def perform_text_analysis():
	#open metadata file
	metadata_filepath = os.path.join(ROOT_DIR, METADATA_FILENAME)
	print("Opening metadata")
	with open(metadata_filepath, "r") as metadata_fp:
		
		#create csv reader object for metadata file
		metadata_reader = csv.reader(metadata_fp, delimiter='\t')
		
		#initialize data structures and vars for metadata iteration
		index_table = {} #column index of each keyword
		text_ids = {} #text ids partitioned by topic and decade
		primary_topics = [] #list of topics
		header_row = True
		# use metadata to partition text ids by decade and primary topic
		for row in metadata_reader:
			if header_row:
				ind = 0
				for header in row:
					index_table[header] = int(ind)
					ind += 1
				header_row = False

			elif row[index_table[TYPE_COL]] == "fla" or row[index_table[TYPE_COL]] == "article":
				text_key = (row[index_table[PRIMARY_TOPIC_COL]], int(row[index_table[DECADE_COL]]))
				if not (row[index_table[PRIMARY_TOPIC_COL]] in primary_topics):
					primary_topics.append(row[index_table[PRIMARY_TOPIC_COL]])
				if not text_ids.get(text_key):
					text_ids[text_key] = []
				text_ids[text_key].append(row[index_table[ID_COL]])
	
	klds = {}
	for topic in primary_topics:
		print(f"Analyzing {topic}")
		klds[topic] = [float("nan")] * (NUM_DECADES - 3)
		for i, decade in enumerate(DECADES[2:-1]):
			#print(f"PRE: {DECADES[i+1]} | POST: {decade}")
			pre_decade = DECADES[i+1]
			pre_wordcount = 0
			pre_key = (topic, pre_decade)
			pre_lemmamap = {} # maps of lemma to its # of occurrences
			post_lemmamap = {}
			for pre_textid in text_ids.get(pre_key, []):
				text_filepath = os.path.join(ROOT_DIR, TEXT_DIR, TEXT_BASE_FILENAME.format(pre_textid))
				with open(text_filepath, "r") as textfile:
					#print(f"Generating soup for textid {pre_textid}")
					text_soup = BeautifulSoup(textfile, 'html.parser')
					text = text_soup.find('text', {'n': pre_textid})
					lemma_tags = text.find_all(lambda t: t.name == 'w' and t.has_attr('lemma'))
					for lemma_tag in lemma_tags:
						lemma = lemma_tag['lemma']
						if lemma and (not lemma in COMMON_WORDS) and len(lemma) >= MIN_LEMMA_LEN:
							pre_wordcount += 1
							if not pre_lemmamap.get(lemma):
								pre_lemmamap[lemma] = 1
								post_lemmamap[lemma] = 0
							else:
								pre_lemmamap[lemma] += 1

			# repeat for post period
			post_decade = decade
			post_key = (topic, post_decade)
			post_wordcount = 0
			for post_textid in text_ids.get(post_key, []):
				text_filepath = os.path.join(ROOT_DIR, TEXT_DIR, TEXT_BASE_FILENAME.format(post_textid))
				with open(text_filepath, "r") as textfile:
					#print(f"Generating soup for textid {post_textid}")
					text_soup = BeautifulSoup(textfile, 'html.parser')
					text = text_soup.find('text', {'n': post_textid})
					lemma_tags = text.find_all(lambda t: t.name == 'w' and t.has_attr('lemma'))
					for lemma_tag in lemma_tags:
						lemma = lemma_tag['lemma'].lower()
						if lemma and (not lemma in COMMON_WORDS) and len(lemma) >= MIN_LEMMA_LEN and lemma.isalpha():
							post_wordcount += 1
							if not pre_lemmamap.get(lemma):
								pre_lemmamap[lemma] = 0
								post_lemmamap[lemma] = 1
							else:
								post_lemmamap[lemma] += 1

			#print(f"PRE WC: {pre_wordcount} | POST WC: {post_wordcount}")

			kld = float("nan")

			if pre_wordcount > 0 and post_wordcount > 0:
				keys = list(pre_lemmamap.keys())
				for k in keys:
					if pre_lemmamap[k] == 0 or post_lemmamap[k] == 0:
						del pre_lemmamap[k]
						del post_lemmamap[k]
						# pre_wordcount -= 1
						# post_wordcount -= 1

				filter_threshold = INITIAL_FREQUENCY_DIFF_FILTER
				pre_lemmamap_filtered = pre_lemmamap.copy()
				post_lemmamap_filtered = post_lemmamap.copy()
				pre_lemma_freqs = []
				post_lemma_freqs = []
				for k in pre_lemmamap:
					if (post_lemmamap[k] / post_wordcount * 1000000) - (pre_lemmamap[k] / pre_wordcount * 1000000) < filter_threshold:
						del pre_lemmamap_filtered[k]
						del post_lemmamap_filtered[k]
					else:
						pre_lemma_freqs.append(pre_lemmamap[k] / pre_wordcount * 1000000)
						post_lemma_freqs.append(post_lemmamap[k] / post_wordcount * 1000000)
						#print(f"PRE FREQUENCY OF {k} is {pre_lemmamap[k] / pre_wordcount * 1000000}")
						#print(f"POST FREQUENCY OF {k} is {post_lemmamap[k] / post_wordcount * 1000000}")

				p_value = stats.ttest_ind(pre_lemma_freqs, post_lemma_freqs, equal_var=False).pvalue

				#print(f"INITIAL P VALUE: {p_value}")
				best_threshold = filter_threshold

				while ((p_value > 0.01 or p_value != p_value) and filter_threshold > 1):
					filter_threshold *= 0.75
					#print(f"THRESHOLD: {filter_threshold}")
					pre_lemmamap_filtered = pre_lemmamap.copy()
					post_lemmamap_filtered = post_lemmamap.copy()
					pre_lemma_freqs_attempt = []
					post_lemma_freqs_attempt = []
					for k in pre_lemmamap:
						if (post_lemmamap[k] / post_wordcount * 1000000) - (pre_lemmamap[k]/ pre_wordcount * 1000000 ) < filter_threshold:
							del pre_lemmamap_filtered[k]
							del post_lemmamap_filtered[k]
						else:
							#print(f"PRE FREQUENCY OF {k} is {pre_lemmamap[k] / pre_wordcount * 1000000}")
							#rint(f"POST FREQUENCY OF {k} is {post_lemmamap[k] / pre_wordcount * 1000000}")
							pre_lemma_freqs_attempt.append(pre_lemmamap[k] * 1000000 / pre_wordcount)
							post_lemma_freqs_attempt.append(post_lemmamap[k] * 1000000 / post_wordcount)

					p_value_new = stats.ttest_ind(pre_lemma_freqs_attempt, post_lemma_freqs_attempt, equal_var=False).pvalue
					if p_value_new < p_value or p_value != p_value:
						p_value = p_value_new
						pre_lemma_freqs = pre_lemma_freqs_attempt
						post_lemma_freqs = post_lemma_freqs_attempt
						best_threshold = filter_threshold
					#print(f"ITERATIVE P VALUE: {p_value}")

				#print(f"FINAL THRESHOLD: {best_threshold} | NUM WORDS: {len(pre_lemma_freqs)}")
				kld = float("nan")
				if len(pre_lemma_freqs) > 0 and p_value < 0.05:
					presum = sum(pre_lemma_freqs)
					postsum = sum(post_lemma_freqs)
					freq4 = [x/presum for x in pre_lemma_freqs]
					freq4p = [x/postsum for x in post_lemma_freqs]
					kld = sum([x * math.log(x, 2) for x in freq4p]) - sum(x * math.log(freq4[i], 2) for i, x in enumerate(freq4p))
			#print(f"KLD FOR TOPIC: {topic} FOR DECADE: {decade} IS {kld}")
			klds[topic][i] = kld
	relative_entropy_output_filepath = os.path.join(ROOT_DIR, RELATIVE_ENTROPY_OUTPUT_FILENAME)
	with open(relative_entropy_output_filepath, "w") as entropy_outfile:
		entropy_writer = csv.writer(entropy_outfile)
		entropy_writer.writerow(["Decades"] + DECADES[2:-1])
		for topic in klds:
			entropy_writer.writerow([topic] + [kld if kld == kld else "" for kld in klds[topic]])
	plot_entropies_from_file()


def plot_entropies_from_file():
	output_filepath = os.path.join(ROOT_DIR, RELATIVE_ENTROPY_OUTPUT_FILENAME)
	if not os.path.exists(output_filepath):
		print(f"ERROR: NO file in {ROOT_DIR} named {RELATIVE_ENTROPY_OUTPUT_FILENAME}, make sure ROOT_DIR and RELATIVE_ENTROPY_OUTPUT_FILENAME are properly set, and make sure you've already run function (1) to produce the output")
		return
	
	with open(output_filepath, "r") as entropy_file:
		entropy_reader = csv.reader(entropy_file)
		first = True
		corr_lst = []
		for row in entropy_reader:
			if first:
				first = False
				continue
			
			topic = row[OUTPUT_FILE_TOPIC_COL]
			data = [float(x) if x else float('nan') for x in row[OUTPUT_FILE_TOPIC_COL+1:]]
			plt.figure()
			plt.title(f"Relative Entropy vs Time for Topic: {topic}")
			plt.xlabel("Decade")
			plt.ylabel("Relative Entropy")
			linreg_result = stats.linregress([DECADES[x+2] for x in range(NUM_DECADES - 3) if data[x] == data[x]], [kld for kld in data if kld == kld])
			plt.plot([DECADES[x+2] for x in range(NUM_DECADES - 3) if data[x] == data[x]], [kld for kld in data if kld == kld], label='Data', linestyle='-', marker='o')
			plt.plot([DECADES[x+2] for x in range(NUM_DECADES - 3) if data[x] == data[x]], [x*linreg_result.slope + linreg_result.intercept for x in [DECADES[x+2] for x in range(NUM_DECADES - 3) if data[x] == data[x]]], label=f"Linreg: y = {linreg_result.slope:.5f}x + {linreg_result.intercept:.3f} (r = {linreg_result.rvalue:.3f})", linestyle='-')
			plt.legend()
			plt.savefig(os.path.join(ROOT_DIR, PLOT_BASE_FILENAME.format(topic)), bbox_inches='tight')
			corr_lst.append((topic, linreg_result.rvalue))
		corr_lst.sort(key=lambda t: abs(t[1]))
		print(corr_lst)
		plt.show()





if __name__ == "__main__":
	repeat = True
	while repeat:
		user_input = input(
			"Choose function: topic analysis (0), text analysis (1), plot entropies from data (2), quit (3)\n")
		if user_input == "0":
			perform_topic_analysis()
		if user_input == "1":
			perform_text_analysis()
		if user_input == "2":
			plot_entropies_from_file()
		if user_input == "3":
			repeat = False
	print("SUCCESS")
