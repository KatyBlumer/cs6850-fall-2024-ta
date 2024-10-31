The autograder works by comparing each student's answers to the average answer across the entire class. (Disclaimer: I didn't write the code, this is my understanding of it after reading through it.)

More specifically, here's how it works for each question.

## Questions 1 and 2
Questions 1 and 2 are each scored in 2 parts. First, we calculate the mean of each line in the answer across all students.
Then for each individual student, we calculate the [Wasserstein distance](https://en.wikipedia.org/wiki/Wasserstein_metric) 
(earth mover distance) between their answer and the class's average answer. 
The Wasserstein distance is the first number that's scored for each question.
The other number is (for Q1) the total number of edges or (for Q2) the size of the largest connected component. 
For each question, the two numbers are Z-scored across the class. 
A Z-score is the number of standard deviations from the class's mean.
Each Z-score is assigned 3 points if it's less than 1, 2 points if it's between 1 and 2, and 1 point if greater than 2.
The two Z-scores are averaged for each question to give a total score from 1-3.

(It might seem odd to Z-score the Wasserstein distance, since a lower distance from the average distribution is 
should always be better than a higher one. But in practice, no one's distance is anywhere near a full standard deviation 
closer to 0 than the mean distance.)

## Questions 3 and 4

For questions 3 and 4, we calculate the mean and standard deviation over the whole class for each line of the question. 
Then for each student, we calculate how many standard deviations from the mean they are on every line. 
We give 3 points for answers within 1 standard deviation from the mean, 2 points if within 1-2 standard deviations, 
and 1 point if more than 2 standard deviations. 
Then we average across all lines in the question to get a score from 1-3 for each question.


As an example, here's a potential answer for problem 3:

<img width="100" alt="image" src="https://github.com/user-attachments/assets/136b6831-8704-4bea-b5c6-fbe43629876a">


And here are the means and standard deviations across the entire class:

<img width="172" alt="image" src="https://github.com/user-attachments/assets/eb5f4be1-ecf5-49c6-a138-fb6edfc9576e">

Our sample answer is within 1 standard deviation of the mean for every line except for 2 and 3. 
So their scores for each line would be: [3, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3]. 
Their overall score for question 3 would be the average of those, or 2.82.
