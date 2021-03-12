# AUTOGENERATED! DO NOT EDIT! File to edit: notebooks/Modelling.ipynb (unless otherwise specified).

__all__ = ['Sayer', 'sy']

# Cell
class Sayer:
    def __init__(self,name=''):
        self.name = name

    def say_hello(self):
        return(f'say hello to {self.name}')

# Cell
import pandas as pd

# Cell
sy = Sayer('puneet')
assert sy.say_hello() == 'say hello to puneet'