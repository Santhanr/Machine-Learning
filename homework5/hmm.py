from __future__ import print_function
import json
import numpy as np
import sys

def forward(pi, A, B, O):
  """
  Forward algorithm

  Inputs:
  - pi: A numpy array of initial probailities. pi[i] = P(Z_1 = s_i)
  - A: A numpy array of transition probailities. A[i, j] = P(Z_t = s_j|Z_t-1 = s_i)
  - B: A numpy array of observation probabilities. B[i, k] = P(X_t = o_k| Z_t = s_i)
  - O: A list of observation sequence (in terms of index, not the actual symbol)

  Returns:
  - alpha: A numpy array alpha[j, t] = P(Z_t = s_j, x_1:x_t)
  """
  S = len(pi)
  N = len(O)
  alpha = np.zeros([S, N])
  #print(alpha.shape,A.shape,pi.shape,B.shape)
  ###################################################
  # Q3.1 Edit here
  ###################################################
  for j in range(0,alpha.shape[0]):
    alpha[j][0]=pi[j]*B[j][O[0]]
  
  for t in range(1,alpha.shape[1]):
    for j in range(0,alpha.shape[0]):
        tSum=0
        for i in range(A.shape[0]):
            tSum+=alpha[i][t-1]*A[i][j]
        alpha[j][t]=B[j][O[t]]*tSum
  return alpha


def backward(pi, A, B, O):
  """
  Backward algorithm

  Inputs:
  - pi: A numpy array of initial probailities. pi[i] = P(Z_1 = s_i)
  - A: A numpy array of transition probailities. A[i, j] = P(Z_t = s_j|Z_t-1 = s_i)
  - B: A numpy array of observation probabilities. B[i, k] = P(X_t = o_k| Z_t = s_i)
  - O: A list of observation sequence (in terms of index, not the actual symbol)

  Returns:
  - beta: A numpy array beta[j, t] = P(Z_t = s_j, x_t+1:x_T)
  """
  S = len(pi)
  N = len(O)
  beta = np.zeros([S, N])
  ###################################################
  # Q3.1 Edit here
  ###################################################
  
  for j in range(0,beta.shape[0]):
    beta[j][N-1]=1
  
  for t in range(beta.shape[1]-1,0,-1):
    for j in range(0,beta.shape[0]):
        tSum=0
        for i in range(A.shape[0]):
            tSum+=beta[i][t]*A[j][i]*B[i][O[t]]
        beta[j][t-1]=tSum  
  return beta

def seqprob_forward(alpha):
  """
  Total probability of observing the whole sequence using the forward algorithm

  Inputs:
  - alpha: A numpy array alpha[j, t] = P(Z_t = s_j, x_1:x_t)

  Returns:
  - prob: A float number of P(x_1:x_T)
  """
  prob = 0
  ###################################################
  # Q3.2 Edit here
  ###################################################
  for i in range(alpha.shape[0]):
   prob = prob + alpha[i,alpha.shape[1]-1]
  return prob


def seqprob_backward(beta, pi, B, O):
  """
  Total probability of observing the whole sequence using the backward algorithm

  Inputs:
  - beta: A numpy array beta: A numpy array beta[j, t] = P(Z_t = s_j, x_t+1:x_T)
  - pi: A numpy array of initial probailities. pi[i] = P(Z_1 = s_i)
  - B: A numpy array of observation probabilities. B[i, k] = P(X_t = o_k| Z_t = s_i)
  - O: A list of observation sequence
      (in terms of the observation index, not the actual symbol)

  Returns:
  - prob: A float number of P(x_1:x_T)
  """
  prob = 0
  ###################################################
  # Q3.2 Edit here
  ###################################################
  for i in range(beta.shape[0]):
   prob = prob + beta[i,0]*pi[i]*B[i,O[0]]
  return prob

def viterbi(pi, A, B, O):
  """
  Viterbi algorithm

  Inputs:
  - pi: A numpy array of initial probailities. pi[i] = P(Z_1 = s_i)
  - A: A numpy array of transition probailities. A[i, j] = P(Z_t = s_j|Z_t-1 = s_i)
  - B: A numpy array of observation probabilities. B[i, k] = P(X_t = o_k| Z_t = s_i)
  - O: A list of observation sequence (in terms of index, not the actual symbol)

  Returns:
  - path: A list of the most likely hidden state path k* (in terms of the state index)
    argmax_k P(s_k1:s_kT | x_1:x_T)
  """
  path = []
  ###################################################
  # Q3.3 Edit here
  ###################################################
  S = len(pi)
  N = len(O)
  w = np.zeros([S+2, N])
  trackback = np.zeros([S+2,N])
  
  for j in range(0,S):
    w[j][0]=pi[j]*B[j][O[0]]
  
  for t in range(1,N):
    for j in range(0,S):
        tSum=w[0][t-1]*A[0][j]*B[j][O[t]]
        index=0
        for i in range(1,S):
            if(w[i][t-1]*A[i][j]*B[j][O[t]]>tSum):
                tSum=w[i][t-1]*A[i][j]*B[j][O[t]]
                index=i
        w[j][t]=tSum
        trackback[j][t] = index
  
  temp=w[0][N-1]
  index=0
  for i in range(1,S):
    if(w[i][N-1]>temp):
        temp=w[i][N-1]
        index=i  
  
  w[S+1,N-1] = temp
  trackback[S+1,N-1]=index
   
  node = int(trackback[S+1][N-1])
  path.append(node)
  for i in range(N-1,0,-1):
    node=int(trackback[node][i])
    path.append(node)
  return path


##### DO NOT MODIFY ANYTHING BELOW THIS ###################
def main():
  model_file = sys.argv[1]
  Osymbols = sys.argv[2]

  #### load data ####
  with open(model_file, 'r') as f:
    data = json.load(f)
  A = np.array(data['A'])
  B = np.array(data['B'])
  pi = np.array(data['pi'])
  #### observation symbols #####
  obs_symbols = data['observations']
  #### state symbols #####
  states_symbols = data['states']

  N = len(Osymbols)
  O = [obs_symbols[j] for j in Osymbols]

  alpha = forward(pi, A, B, O)
  beta = backward(pi, A, B, O)

  prob1 = seqprob_forward(alpha)
  prob2 = seqprob_backward(beta, pi, B, O)
  print('Total log probability of observing the sequence %s is %g, %g.' % (Osymbols, np.log(prob1), np.log(prob2)))

  viterbi_path = viterbi(pi, A, B, O)

  print('Viterbi best path is ')
  for j in viterbi_path:
    print(states_symbols[j], end=' ')

if __name__ == "__main__":
  main()
