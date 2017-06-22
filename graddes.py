# numpy - matrix multiplication library
# Hyperparameters in ML - are tuning knowbs for the model. They are guessed by trial and error. Eg: learning rate.

from numpy import *

# Calculate sum of squared error for one time step (corresponding to one line):
# E = 1/N * (sum1toN(y - (mx + b)^2))
def compute_error_for_given_points(b, m, points):
 totalError = 0
 for i in range(0, len(points)):
  x = points[i,0]
  y = points[i,1]
  totalError += (y - (m * x + b)) **2 # ** is power operator
  return totalError / float(len(points))

def step_gradient(b_current, m_current, points, learning_rate):
 b_gradient = 0
 m_gradient = 0
 N = float(len(points))
 for i in range(0, len(points)):
  x = points[i,0]
  y = points[i,1]
  b_gradient += -(2/N) * (y - ((m_current * x) + b_current))
  m_gradient += -(2/N) * x * (y - ((m_current * x) + b_current)) 
 new_b = b_current - (learning_rate * b_gradient)
 new_m = m_current - (learning_rate * m_gradient)
 return (new_b, new_m)

def gradient_descent_runner(points, starting_b, starting_m, learning_rate, num_iterations):
 b = starting_b
 m = starting_m
 for i in range(num_iterations):
  b, m = step_gradient(b,m, array(points), learning_rate)
  return [b,m]

def run():
 points = genfromtext('data.csv', delimiter = ',')
 learning_rate = 0.0001
 initial_b = 0 #y-intercept
 initial_m = 0 #slope
 num_iter = 1000
 [b,m] = gradient_descent_runner(points, initial_b, initial_m, learning_rate, num_iter)
 print(b,m)

if__name__ = '__main__':
 run()