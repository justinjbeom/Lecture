import numpy as np

# Return a single sample sampled from a negative binomial NB(r,p) distribution.
def sample_nb(r, p):
	# Your code here
	return sample

# Execute monte carlo to determine E[x] where x = NB(r,p). Return the resulting value.
def montecarlo(r, p, N):
	# Your code here
	return result

if __name__ == "__main__":
	# Use this section to execute and check your implementation.
	# This section is not used when grading.
	print("Calculate the mean of NB(3, 0.7).")
    mean = montecarlo(3, 0.7, 100)
    print("Mean value approximated with {0} samples: {1}".format(N, mean))