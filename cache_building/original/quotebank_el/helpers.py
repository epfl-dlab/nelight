from imports import *
import gzip

os.environ['JAVA_HOME'] = '/usr/lib/jvm/java-11-openjdk-amd64/'
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)


def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def save_pickle(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


reverse_dict = lambda x: {v: k for k, v in x.items()}

from nltk.corpus import stopwords
sw = set(stopwords.words('english'))
sw.add("'re")
sw.add("n't")
sw.add("'s")
sw.add("'ve")
sw.add("'ll")

special_characters = {'.', ',', ')', '(', '/', '\\', '<', '>', '+', '-', '!', '?', '@', '"', '#', '$', '%', '&', '=', '[', ']', '{', '}', "'", '~', '*', '_', ':'}


def ignore_warnings():
    warnings.filterwarnings('ignore')

def _open(file, io, compression=gzip):
    if compression is None:
        return open(file, io)
    return compression.open(file, io)

def qtoi(qid):
    return int(qid[1:])


def pd_to_nk(df, directed=False, weighted=False):
    nid_to_qid = dict(enumerate(set(df.source.unique()).union(df.target.unique())))
    qid_to_nid = reverse_dict(nid_to_qid)
    G = nk.Graph(n=len(nid_to_qid), directed=directed, weighted=weighted)

    for index, edge in df.iterrows():
        source = qid_to_nid[edge[0]]
        target = qid_to_nid[edge[1]]
        if weighted:
            G.addEdge(source, target, edge[2])
        else:
            G.addEdge(source, target)

    return G, nid_to_qid


def start_spark(config=None, appName='Quotegraph', n_threads=24):
    spark = (SparkSession.builder.master(f'local[{n_threads}]')
             .appName(appName)
             .config("spark.driver.memory", "32g")
             .config("spark.executor.memory", "32g")
             .config('spark.local.dir', '/dlabdata1/culjak/tmp')
             .config('spark.sql.execution.arrow.pyspark.enabled', 'true'))
    if config is not None:
        for k, v in config:
            spark = spark.config(k, v)

    return spark.getOrCreate()


@F.udf(T.StringType())
def first(arr):
    return arr[0] if arr is not None else None


analyzer = SentimentIntensityAnalyzer()


@F.udf(T.FloatType())
def vader_sentiment(quotation):
    return analyzer.polarity_scores(quotation)['compound']


@F.udf(T.StringType())
def quoteID2date(quoteID):
    return quoteID[:10]


party_switches = pd.read_csv('/dlabdata1/culjak/wikidata_knowledge/quotebank_party_switches.csv').sort_values('switch_date')

party_switch_events = {}
for row in party_switches.iterrows():
    qid, from_party, to_party, switch_date = row[1]
    if qid not in party_switch_events:
        party_switch_events[qid] = []
    party_switch_events[qid].append([from_party, to_party, switch_date])



@F.udf(T.StringType())
def get_party_udf(qid, party, quoteID=None, date=None):
    if party is None:
        return None
    if len(party) == 1:
        return party[0]

    if date is None:
        date = quoteID[:10]
    if qid not in party_switch_events:
        return party[-1]

    switches = party_switch_events[qid]

    if len(switches) == 0:
        return party[-1]

    first_switch = switches[0]
    if date < first_switch[2]:
        return first_switch[0]

    for event in switches:
        if event[2] < date:
            return event[1]
