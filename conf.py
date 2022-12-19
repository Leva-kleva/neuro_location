import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix


path_to_data = "static/worldcities.csv"
