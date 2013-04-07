
from lgSmarts.sklearnNn import nn_model

ins = [ [ 0, 1 ], [ 1, 1 ], [ 2, 2 ] ]
outs = [ [0.0], [0.5], [0.75] ]

m = nn_model()
m.fit(ins, outs)
for i in ins:
    print(m.predict(i))
import IPython
IPython.embed()

