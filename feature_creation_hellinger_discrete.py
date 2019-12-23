import numpy as np
import scipy.stats as ss

import operator
import utils.file_ops
import csv

from deap import base
from deap import creator
from deap import tools
from deap import gp
from deap import algorithms

from tqdm import tqdm


def protect_overflow(f):
    min_value = -1e15
    max_value = 1e15

    def wrapper(*args, **kwargs):
        x = f(*args, **kwargs)

        return max(min(x, max_value), min_value)

    return wrapper


def protectedDiv(x, y):
    if y == 0:
        return 1
    return x / y


def log_safe(x):
    if x <= 0:
        return -2
    return np.log(x)


# List of Operators and Constants
pset = gp.PrimitiveSet('main', 36)
pset.addPrimitive(protect_overflow(operator.add), 2, name='add')
pset.addPrimitive(protect_overflow(operator.sub), 2, name='sub')
pset.addPrimitive(protect_overflow(operator.mul), 2, name='mul')
pset.addPrimitive(protect_overflow(protectedDiv), 2, name='div')
# pset.addPrimitive(lambda x, y: int(operator.lt(x, y)), 2, name='lt')
# pset.addPrimitive(lambda x, y: int(operator.gt(x, y)), 2, name='gt')
pset.addPrimitive(operator.neg, 1, name='-')
pset.addPrimitive(max, 2, name='max')
pset.addPrimitive(min, 2, name='min')
pset.addPrimitive(np.cos, 1, name='cos')
pset.addPrimitive(np.sin, 1, name='sin')
pset.addPrimitive(protect_overflow(log_safe), 1, name='log')
# pset.addPrimitive(np.tan, 1)
pset.addPrimitive(np.abs, 1, name='abs')
pset.addTerminal(-1)
pset.addEphemeralConstant('rand_int', lambda: np.random.randint(-10, 10))
pset.addEphemeralConstant('rand_float', lambda: np.random.rand())

creator.create('FitnessMax', base.Fitness, weights=(1.0,))
creator.create('Individual', gp.PrimitiveTree, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register('expr', gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
toolbox.register('individual', tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register('population', tools.initRepeat, list, toolbox.individual)
toolbox.register('compile', gp.compile, pset=pset)

def create_new_features(X, Y, n=4):
    def hellinger(individual, X, Y):
        func = toolbox.compile(expr=individual)

        X0 = X[Y==-1]
        X1 = X[Y==1]

        genX0 = np.array([func(*X0[i]) for i in range(X0.shape[0])])
        genX1 = np.array([func(*X1[i]) for i in range(X1.shape[0])])

        genX = np.array([func(*X[i]) for i in range(X.shape[0])])

        # make densities
        u, c = np.unique(genX0, return_counts=True)
        c = c / np.sum(c)
        k0 = np.vectorize(lambda x: 0 if not np.any(u == x) else c[u == x])

        u, c = np.unique(genX1, return_counts=True)
        c = c / np.sum(c)
        k1 = np.vectorize(lambda x: 0 if not np.any(u == x) else c[u == x])

        p_x0 = k0(genX)
        p_x0 = p_x0 / np.sum(p_x0)
        p_x1 = k1(genX)
        p_x1 = p_x1 / np.sum(p_x1)
        # p_y0 = X0.shape[0] / (X0.shape[0] + X1.shape[0])
        # p_y1 = X1.shape[0] / (X0.shape[0] + X1.shape[0])
        # p_x = (p_y0 * p_x0) + (p_y1 * p_x1)

        # p_y0x = (p_x0 * p_y0) / p_x
        # p_y1x = (p_x1 * p_y1) / p_x

        # dist = np.sqrt(np.sum(np.square(np.sqrt(p_y0x) - np.sqrt(p_y1x))))
        dist = np.sqrt(np.sum(np.square(np.sqrt(p_x0) - np.sqrt(p_x1))))

        return dist,

    toolbox.register('evaluate', hellinger, Y=Y, X=X)
    toolbox.register('select', tools.selTournament, tournsize=3)
    toolbox.register('mate', gp.cxOnePoint)
    toolbox.register('expr_mut', gp.genFull, min_=0, max_=2)
    toolbox.register('mutate', gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

    toolbox.decorate('mate', gp.staticLimit(key=operator.attrgetter('height'), max_value=17))
    toolbox.decorate('mutate', gp.staticLimit(key=operator.attrgetter('height'), max_value=17))

    pop = toolbox.population(n=300)
    hof = tools.HallOfFame(n)

    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register('avg', np.mean)
    mstats.register('std', np.std)
    mstats.register('min', np.min)
    mstats.register('max', np.max)

    pop, log = algorithms.eaSimple(pop, toolbox, 0.8, 0.1, 20, stats=mstats, halloffame=hof, verbose=True)

    features_expr = [str(gp.PrimitiveTree(ind)) for ind in hof]
    features_func = [toolbox.compile(expr=ind) for ind in hof]

    return features_expr, features_func, str(log)


exploits = [
    'freak', 'poodle', 'nginx_keyleak', 'nginx_rootdir', 'logjam',
    # 'nginx_rootdir', 'logjam',
    'orzhttpd_rootdir', 'orzhttpd_restore'
]

logs = []

for exploit in tqdm(exploits):
    X, Y = utils.file_ops.load_data(
        './data/raid/{}_data.csv'.format(
            exploit
        )
    )

    Xt, Yt = utils.file_ops.load_data(
        # './data/raid/{}_data.csv'.format(
        './data/raid/{}/subset_0/test_set.csv'.format(
            exploit
        )
    )

    a = X.min(axis=0)
    b = X.max(axis=0)
    X = utils.file_ops.rescale(X, a, b, -1, 1)
    Xt = utils.file_ops.rescale(Xt, a, b, -1, 1)

    features_expr, features_func, log = create_new_features(Xt, Yt)

    logs.append(log)

    # with open('./data/raid/{}_hell_gt_lt.csv'.format(exploit), 'w') as f:
    with open('./data/raid/{}_hell_discrete.csv'.format(exploit), 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['#'] + features_expr)

        for i, x in enumerate(X):
            writer.writerow([Y[i]] + [f(*x) for f in features_func])


# with open('log') as f:
    # f.write('\n'.join(logs))
