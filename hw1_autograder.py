# Adapted from papachristoumarios/cs6850-spring-2023-ta
#    (couldn't fork repo since it also contains HW solutions)
import os
import numpy as np
import glob
import collections
from scipy.stats import chi2
from scipy.stats import wasserstein_distance
import pandas as pd
import matplotlib.pyplot as plt

submission_dir = "submissions" #@param
grade_output_file = "grades_HW1.csv" #@param


def print_nice_arr(arr):
  with np.printoptions(precision=3, suppress=True, threshold=10000, linewidth=10000):
    print(arr)

def read_files(submission_dir):
  filelist = glob.glob(f'{submission_dir}/*hw1solution*.txt')

  all_versions = collections.defaultdict(list)

  # Dict of student name to all files with student name
  for filename in filelist:
        basename = os.path.basename(filename)
        canvas_id = '_'.join(basename.split('_')[0:2])#filename.split('_')[1]
        all_versions[canvas_id].append(filename)

  # Get latest file version for each student (some uploaded multiple times)
  #   latest_versions is a dict of studentname : latest version's filename
  latest_versions = {}
  for canvas_id, filenames in all_versions.items():
        if len(filenames) == 1:
            latest_versions[canvas_id] = filenames[0]
        else:
            mx, argmax = -1, -1
            for filename in filenames:
                if os.path.splitext(filename)[0].endswith(r'-\d') and int(os.path.splitext(filename)[0][-1]) > mx:
                    argmax = filename
            latest_versions[canvas_id] = argmax

  # Map of student name / netid to anonymized counter and vice versa
  #  (netid is first line in file, student name is in filename)
  student2idx = {}
  idx2student = {}
  idx2filename = {}
  file_contents = {}
  line_contents = {}  # TODO temp

  counter = 0

  # Open each file, add netid (first line) & "anonymized" counter to dicts; add rest of lines to file_contents dict
  for canvas_id, filename in latest_versions.items():
        with open(filename) as f:
            file_content = f.read().splitlines()
        netid = file_content[0].lower()

        student2idx[canvas_id, netid] = counter
        idx2student[counter] = (canvas_id, netid)
        idx2filename[counter] = (filename)
        file_contents[counter] = file_content[1:]

        counter += 1
  print(f'Processed netids for {counter} students.')

  # Read each student's file_contents into answers dict. Save errors in 'errors'.
  #  Values of "answers" are, for each student, a dict of question number: list of answers.
  answers = {}
  errors = ['## PARSING ERRORS ##']

  for subm_id, file_content in file_contents.items():
    answers[subm_id] = collections.defaultdict(list)
    canvas_id, netid = idx2student[subm_id]
    line_contents[subm_id] = []

    for line in file_content:
      if not line.startswith('@'):
        continue
      if '\t' in line:
        line = line.replace('\t', ' ')
      line = line.rstrip(' ')
      line = line.split(' ')

      if line[1] not in ['1', '2', '3', '4']:
        errors.append(f'Subm id: {subm_id} Name: {canvas_id}, NetID: {netid}, Line: {line} Error 0: unrecognized question number {line[1]}')
        continue
      ques_id = int(line[1])

      line_contents[subm_id].append(line)

      if (line == ['@', '2', '0', '0']) or (line == ['@', '3', '0', '1']):
        # Some students start at index 0 and some start at 1 for Q2-3.
        # For Q2, this will overwrite the LCC ratio, since the code uses line[2] as an index into aggregate_answers.
        # For Q3, it just messes up the Z-score, since other students will have the default aggregate_answers value of 0, so causes the student to lose points,
        #  which could maybe be fair since it doesn't follow the example given in the pset, but... why not fix it.
        # Not an issue for Q1 or Q4: Q1 because 0 should be included, and Q4 because the correct value for index 0 is 0, so it will be the same for students that didn't include that line.
        continue

      if len(line) == 4:
          try:
              x, y = float(line[2]), float(line[3])
              answers[subm_id][ques_id].append([x, y])
          except Exception as e:
              canvas_id, netid = idx2student[subm_id]
              if ',' in line[2]: # one student had commas in Q2 response
                line[2] = line[2].replace(',', '')
                x, y = float(line[2]), float(line[3])
                answers[subm_id][ques_id].append([x, y])
              else:
                errors.append(f'Subm id: {subm_id} Name: {canvas_id}, NetID: {netid}, Question: {ques_id}, Line: {line}, Error A: {e}')
      elif ques_id == 2 and len(line) > 4:  # First line for Q2 has extra values; get ratio reported at end of line
          try:
              x, y = 0, float(line[-1])
              answers[subm_id][ques_id].append([x, y])
          except Exception as e:
            x, y = 0, float(line[-1])
            errors.append(f'Subm id: {subm_id} Name: {canvas_id}, NetID: {netid}, Question: {ques_id} (LCC), Line: {line}, Line[-1]: {line[-1]}, Error B: {e}')
      else:
        canvas_id, netid = idx2student[subm_id]
        errors.append(f'Subm id: {subm_id} Name: {canvas_id}, NetID: {netid}, Question: {ques_id}, Line: {line}, Error C: Line wrong length ({len(line)})')

  # Get maximum values of "j" (degree) for each Q.
  #   Iterates through students; for each question, student's max answer is compared to running maximum.
  #   This tells us the max length of answers for each question.
  max_range = collections.defaultdict(lambda: -1)

  for subm_id in answers.keys():
      for ques_id in answers[subm_id].keys():
          answers[subm_id][ques_id] = np.array(answers[subm_id][ques_id], dtype=np.float64)
          max_range[ques_id] = max(max_range[ques_id], np.max(answers[subm_id][ques_id][:, 0]))


  # Create matrices for each question, with shape [num_students, m_rng]
  aggregate_answers = {}

  errors.append('')
  errors.append('## MATRIX PARSING ERRORS ##')

  for ques_id, m_rng in max_range.items():
      aggregate_answers[ques_id] = np.zeros(shape=(counter, int(m_rng) + 1), dtype=np.float64)

      for subm_id in answers.keys():
          try:
              idx = answers[subm_id][ques_id][:, 0]
              values = answers[subm_id][ques_id][:, 1]
              idx = idx.astype(int)
              aggregate_answers[ques_id][subm_id, idx] = values
          except Exception as e:
              canvas_id, netid = idx2student[subm_id]
              errors.append(f'Subm id: {subm_id} Name: {canvas_id}, NetID: {netid}, Question: {ques_id}, Error MAT: {e}')

  print(f'Finished reading files with {len(errors)-3} errors.')
  return student2idx, idx2student, idx2filename, answers, aggregate_answers, errors, file_contents, line_contents

def score_zscores(Z):
  Z = np.abs(Z)  # Just in case; very important given that we use signed thresholds!
  scores = np.zeros_like(Z)

  # Give 3 points for every answer within 1 std dev of mean
  scores += (3 * (Z <= 1))
  # Give 2 points for every answer between 1-2 std dev of mean
  scores += (2 * (
      (Z > 1)
      &
      (Z <=2)
    ))
  # Give 1 point for every answer greater than 2 std dev from mean
  scores += 1 * (Z > 2)

  return scores

def grade_univariate(X, log_scale=False):
    if log_scale:
        X = np.log(1 + X)

    X_std = X.std(0) + 1e-6
    X_mean = np.mean(X, axis=0)

    # Produce Zscore: # std deviations from mean for each value in the question
    Z = (X - X_mean) / X_std
    Z = np.abs(Z) # Very important given how we threshold below!

    scores = score_zscores(Z)

    # Take mean of scores for each student, to produce a score from 0-3 on the question as a whole.
    scores = scores.mean(-1)

    return scores

# Note: Tried using median instead of mean with Z-score, since mean allows common outliers to throw everything off;
#   it definitely made the Z-score matrix easier to read, but it made total scores lower for some reason? (18 vs 13 studetns who lost points on Q3)
#   Guessing there were 5 people right on the cusp of std dev?
# With mean:
#   Fraction of students with lost points 0.23636363636363636
#   Fraction of students with lost points, per question: [0.03636364 0.13636364 0.11818182 0.09090909]
# With median:
#   Fraction of students with lost points 0.2818181818181818
#   Fraction of students with lost points, per question: [0.03636364 0.13636364 0.16363636 0.09090909]

def find_zscore_and_outliers(Y, idx2student, num_std=1):
    Z = (Y - Y.mean()) / (Y.std() + 1e-6)
    # #TEMP use median instead of mean
    # Z = (Y - np.median(Y)) / (Y.std() + 1e-6)

    outliers = (np.abs(Z) > 1).astype(np.int64)
    outliers_ids = outliers.nonzero()[0]
    outliers_netids = []

    for i in outliers_ids:
        outliers_netids.append((idx2student[i][1], Z[i], int(i)))

    return Z, outliers, outliers_ids, outliers_netids


def score_wasserstein(X, qnum, label="class's mean"):
    # Wasserstein distance (EMD) of distributions of node degree counts
    #   (EMD=Earth Mover Distance, another name for Wasserstein)
    X_emp = X.mean(0)

    dist = np.zeros(X.shape[0], dtype=np.float64)

    for subm_i in range(X.shape[0]):
        dist[subm_i] = wasserstein_distance(X[subm_i, :], X_emp)


    Z_emd_signed, emd_outliers, emd_outlier_ids, emd_outlier_netids = find_zscore_and_outliers(dist, idx2student)
    # In this case, do *not* take the absolute value, since we're working with distances - someone
    # with a negative zscore is *closer* to the average answer distribution than most people in the class,
    # and should not be penalized.
    # (In practice, no one is ever a full standard deviation *below* the mean distance, 
    #   and since no one loses points if they're within a standard deviation of the mean,
    #   it likely won't matter if we use signed or unsigned Z-scores. But we could 
    #   someday change how we do scoring or something, so worth being careful.)
    scores = score_zscores(Z_emd_signed, warn_if_not_abs=False)

    plt.figure()
    plt.title(f"Q{qnum}: Distribution of Wasserstein distances (EMDs) from the class's mean answer")
    sns.histplot(dist)
    plt.savefig(f'q{qnum}_emd_distribution.png')

    plt.figure()
    plt.title(f'Q{qnum}: Z-score (signed) of Wasserstein distances (EMDs) to {label}')
    sns.histplot(Z_emd_signed)
    plt.savefig(f'q{qnum}_emd_zscores.png')

    print(f"Q{qnum}: {len(emd_outlier_netids)} Outliers in EMD:\n{emd_outlier_netids}")

    return scores

def grade_ex_1(X, idx2student):

    # Total number of edges test (sum degree of all nodes)
    num_edges = X.sum(-1)

    Z_edge_signed, edge_outliers, edge_outlier_ids, edge_outlier_netids = find_zscore_and_outliers(num_edges, idx2student)
    Z_edge_abs = np.abs(Z_edge_signed)


    scores = score_zscores(Z_edge_abs)

    plt.figure()
    plt.title('Q1: Z-score total number of edges')
    counts, bins = np.histogram(Z_edge_signed)
    plt.hist(bins[:-1], bins, weights=counts)

    plt.savefig('q1_total_num_edges.png')

    print(f"Q1: {len(edge_outlier_netids)} Outliers in total nums edges:\n{edge_outlier_netids}")

    scores += score_wasserstein(X, qnum=1, label='mean degree distribution')

    scores /= 2

    print(f'Q1 scores: {scores}')
    print('')

    return scores


def grade_ex_2(X, idx2student):
    # scores = np.zeros(X.shape[0])
    # Number of lcc testsdfg 
    lcc = X[:, 0]

    Z_lcc, lcc_outliers, lcc_outlier_ids, lcc_outlier_netids = find_zscore_and_outliers(lcc, idx2student)
    Z_lcc_abs = np.abs(Z_lcc)

    scores = score_zscores(Z_lcc_abs)

    plt.figure()
    plt.title('Q2: Z-score number of nodes in largest connected component')
    counts, bins = np.histogram(Z_lcc)
    plt.hist(bins[:-1], bins, weights=counts)
    plt.savefig('q2_lcc.png')

    print(f"Q2, {len(lcc_outlier_netids)} Outliers in largest connected component:\n\t{lcc_outlier_netids}")

    scores += score_wasserstein(X, qnum=2, label='connected component sizes')

    scores /= 2

    return scores

def grade_submissions(student2idx, idx2student, answers, aggregate_answers, errors):
    scores = np.zeros(shape=(len(student2idx), len(aggregate_answers)), dtype=np.float64)

    scores[:, 0] = grade_ex_1(aggregate_answers[1], idx2student)
    scores[:, 1] = grade_ex_2(aggregate_answers[2], idx2student)
    scores[:, 2] = grade_univariate(aggregate_answers[3], idx2student)
    scores[:, 3] = grade_univariate(aggregate_answers[4], idx2student)

    grades = scores.sum(-1)

    # print_nice_arr(scores)
    # print_nice_arr(grades)

    print()
    print('Fraction of students with lost points', (grades < 12).sum() / len(grades))
    print(f'Fraction of students with lost points, per question: {(scores < 3).sum(0) / scores.shape[0]}')


    grades_df = []

    for subm_id in range(grades.shape[0]):
        canvas_id, netid = idx2student[subm_id]
        studentname, canvas_id = canvas_id.split('_')
        comment = ', '.join([f'Q{j + 1}:{scores[subm_id, j]:.3f}/3' for j in range(scores.shape[1])])
        grades_df.append([studentname, canvas_id, netid, grades[subm_id], comment])

    columns = ['student_name','canvas_id', 'netid', 'total_score', 'comment']

    grades_df = pd.DataFrame(grades_df, columns=columns)


    grades_df.sort_values(by=['total_score'], inplace=True, ascending=True)

    grades_df.to_csv(grade_output_file, sep=';', index=False)
    # grades_df.to_excel('grades.xlsx')

    with open('errors.txt', 'w+') as f:
        f.write('\n'.join(errors))

    plt.figure(figsize=(10, 5))
    plt.title('Grade distribution for HW1')
    counts, bins = np.histogram(grades)
    plt.hist(bins[:-1], bins, weights=counts)

    plt.savefig('distribution.png')

    return grades_df, scores




student2idx, idx2student, idx2filename, answers, aggregate_answers, errors, file_contents, line_contents = read_files(submission_dir)

print("\nShapes of matrices in aggregate_answers:")
for k, v in aggregate_answers.items():
  print(f"\tQuestion {k}: {v.shape}")'

TEMP = grade_submissions(student2idx, idx2student, answers, aggregate_answers, errors)
