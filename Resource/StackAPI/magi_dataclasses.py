import requests
import time
import json
import re
import os
import sys
import logging
import pickle
import inspect
import pandas as pd
import networkx as nx
import spacy

from itertools import chain
from functools import partial
from bs4 import BeautifulSoup
from typing import List, Tuple, Union, Dict, Callable
from tqdm import tqdm, trange
from datetime import datetime
from pathlib import Path
from github import Github, Repository, UnknownObjectException, RateLimitExceededException
from hn import search_by_date
from markdown import markdown
from requests.exceptions import ConnectionError
from langdetect import detect
from langdetect.detector import LangDetectException
from deep_translator import GoogleTranslator
from deep_translator.exceptions import RequestError as DTRequestError
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import defaultdict

logging.basicConfig(format='%(name)s - %(levelname)s - %(message)s', level=logging.INFO)
magi_dataclasses_logger = logging.getLogger(__name__)

GHV9_TOKENIZED_LINK = 'https://huggingface.co/datasets/Enoch2090/github_semantic_search/resolve/main/ghv9_tokenized.json'

class IdleHandler:
    def __init__(self):
        pass
    
    def github_rate_limit_exceed_idle(self, idle: float = 120):
        magi_dataclasses_logger.info(f'Rate limit exceeded, sleep for {idle}s')
        time.sleep(idle)
    
    def github_rate_limit_control_idle(self, idle: float = 0.1):
        time.sleep(idle)
        
    def translate_rate_exceed_idle(self, idle: float = 60):
        magi_dataclasses_logger.info(f'Translate limit exceeded, sleep for {idle}s')
        time.sleep(idle)
    
@dataclass
class GitHubRepo:
    # data
    name: str = field()
    link: str = field(default = '', repr = False)
    tags: Tuple[str] = field(default = tuple(), repr = False)
    stars: int = field(default = 0)
    description: str = field(default = '')
    lang: str = field(default = '', repr = False)
    repo_lang: str = field(default = '', repr = False)
    readme: str = field(default = '', repr = False)
    hn_comments: str = field(default = '', repr = False)
    gh_updated_time: str = field(default = '', repr = False)
    gh_accessed_time: str = field(default = '', repr = False)
    hn_accessed_time: str = field(default = '', repr = False)
    tokenized_corpus: Tuple[str] = field(default = tuple(), repr = False) 
    
    _fully_initialized: bool = field(default = False)
    
    @property
    def gh_updated_parsed_time(self) -> datetime:
        assert self._fully_initialized, 'Non-fully-initialized GitHubRepo object does not have parsed time'
        return datetime.strptime(self.gh_updated_time, '%Y/%m/%d, %H:%M:%S')
    
    @property
    def gh_accessed_parsed_time(self) -> datetime:
        assert self._fully_initialized, 'Non-fully-initialized GitHubRepo object does not have parsed time'
        return datetime.strptime(self.gh_accessed_time, '%Y/%m/%d, %H:%M:%S')
    
    @property
    def hn_accessed_parsed_time(self) -> datetime:
        assert self._fully_initialized, 'Non-fully-initialized GitHubRepo object does not have parsed time'
        return datetime.strptime(self.hn_accessed_time, '%Y/%m/%d, %H:%M:%S')
    
    @property
    def mentioned_repos(self) -> List[str]:
        assert self._fully_initialized, 'Non-fully-initialized GitHubRepo object does not have mentioned repos'
        pattern = re.compile(r'(https?://)?github.com/(?!{})([^/^\(^\)^\s^<^>^#^\[^\]]*/[^/^\(^\)^\s^<^>^#^\[^\]]*)'.format(self.name))
        return list(set([x[-1] for x in pattern.findall(self.readme)] + [x[-1] for x in pattern.findall(self.hn_comments)]))
    
class GitHubDataset(object):
    MAX_REPO_PER_LANG = 1000
    MIN_STAR_PER_REPO = 50
    TRANSLATE_MAX_RETRY = 3
    CHECKPOINT_PERIOD = 200
    LANG_LIST = ['Python', 'C++', 'JavaScript', 'Rust', 'Go']
    
    def __init__(
        self, 
        empty: bool = True, 
        lang_list: List[str] = None, 
        file_path: str = None, 
        load_nlp: bool = False, 
        load_graph: bool = False,
        nlp_stopword_list: str = './stopwords.txt',
        graph_processors: Dict[str, Callable] = None
    ):
        '''
        Arguments: 
        '''
        self.data = []
        self._load_nlp = load_nlp
        self._load_graph = load_graph
        self._translate_err_counter = 0
        self._it_idx = 0
        if lang_list:
            self.LANG_LIST = lang_list
        self.lang_stats = defaultdict(int)
        self.reverse_map = {}
        self.G = None
        
        if empty:
            return
        if load_nlp:
            with open(nlp_stopword_list, 'r') as f:
                self._stopwords = [x.replace('\n', '') for x in f.readlines()]
        if load_graph and graph_processors is not None:
            for key, func in graph_processors:
                assert type(key) is str, \
                    f'graph_processors argument must be a dict with key type str, got {type(key)}.'
                assert callable(func), \
                    f'graph_processors argument must be a dict with value  a callable type, got {type(key)}, callable=False.'
                assert 'G' in inspect.getfullargspec(func).args, \
                    f'graph_processors argument must be a dict with value  a callable type and having an argument name "G".'
        
        self._init_artifacts(_graph_processors = graph_processors)
        assert file_path, 'GitHubDataset with empty=False initialized with file_path={file_path}'
        if file_path is None:
            # TODO: configure realtime load of link for future updates
            local_file_name = GHV9_TOKENIZED_LINK.split('/')[-1]
            r = requests.get(GHV9_TOKENIZED_LINK, stream=True)
            file_size = int(r.headers.get('content-length'))
            with open(local_file_name, "wb") as f:
                with tqdm(total = file_size // 1024) as _tqdm:
                    chunk_n = 0
                    for chunk in r.iter_content(chunk_size=1024):
                        if chunk:
                            f.write(chunk)
                        chunk_n += 1
                        _tqdm.update(1)
            self.load(GHV9_TOKENIZED_LINK)
        else:    
            self.load(file_path)
        
        if load_graph:
            self._init_interlink_graph()
        
        
    def __getitem__(self, idx):
        assert type(idx) is int or type(idx) is str, 'Index must be either int index or repository name str index.'
        if type(idx) is int:
            return self.data[idx]
        return self.data[self.reverse_map[idx]]
    
    def __setitem__(self, idx, val):
        assert type(val) is GitHubRepo, 'Item must be of GitHubRepo type'
        self.data[idx] = val
    
    def __iter__(self):
        self._it_idx = 0
        return self
        
    def __next__(self):
        if self._it_idx >= len(self.data):
            raise StopIteration
        next_item = self.data[self._it_idx]
        self._it_idx += 1
        return next_item
    
    def __len__(self):
        return len(self.data)
    
    def __contains__(self, repo: Union[str, GitHubRepo]):
        assert type(repo) is str or type(repo) is GitHubRepo, 'Item must be either name str or GitHubRepo type.'
        if type(repo) is str:
            return repo in self.reverse_map.keys()
        return repo.name in self.reverse_map.keys()
        
    def __str__(self):
        langs = ', '.join([f'{lang}: List_{self.lang_stats[lang]}' for lang in self.LANG_LIST])
        return f'GitHubDataset({langs})'
    
    def _remove_code_chunk(self):
        pattern = re.compile('```[\s\S]+```')
        for idx in range(self.__len__()):
            self.data[idx].readme = pattern.sub('\n', self.data[idx].readme)
            self.data[idx].hn_comments = pattern.sub('\n', self.data[idx].hn_comments)
            
    def _remove_html_tags(self):
        pattern = re.compile('<[\s\S]+>')
        for idx in range(self.__len__()):
            self.data[idx].readme = pattern.sub(' ', self.data[idx].readme)
            self.data[idx].hn_comments = pattern.sub(' ', self.data[idx].hn_comments)
    
    def _init_artifacts(
        self, 
        _graph_processors: Dict[str, Callable] = None
    ):
        # see https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/creating-a-personal-access-token
        # I'm tired of defining customed exceptions
        # so if there's nothing, let it be
        self.GH_TOKEN = os.getenv('GH_TOKEN') 
        self._github_artifact = Github(self.GH_TOKEN)
        self._translator_artifact = GoogleTranslator(source='auto', target='en')
        self._idle_handler_artifact = IdleHandler()
        if self._load_nlp:
            self._nlp_artifact = spacy.load('en_core_web_sm')
            self._nlp_artifact.disable_pipe("parser")
            self._nlp_artifact.add_pipe("sentencizer")
            # nlp.max_length of spaCy, diluted since
            # spaces might be inserted
            self._nlp_max_len = int(1000000 * 0.85)
            self._nlp_max_recursion_depth = sys.getrecursionlimit()
            for w in self._stopwords:
                self._nlp_artifact.Defaults.stop_words.add(w)
        if self._load_graph:
            if _graph_processors is not None:
                self._graph_processors_artifacts = _graph_processors
            self._graph_processors_artifacts = {
                'betweenness_centrality': nx.betweenness_centrality,
                'out_degree': lambda G: {n: d for n, d in G.out_degree()},
                'in_degree': lambda G: {n: d for n, d in G.in_degree()},
                'page_rank_0.90': partial(nx.pagerank, alpha=0.90),
                'page_rank_0.50': partial(nx.pagerank, alpha=0.50),
                'hits': lambda G: nx.hits(G)[0]
            }
        
    def _update_repo(
        self, 
        repo_object: GitHubRepo, 
        repo: Repository.Repository = None
    ):
        repo_object.stars = repo.stargazers_count
        if (not repo_object._fully_initialized) or (repo.updated_at > repo_object.gh_updated_parsed_time):
            root_file_list = repo.get_contents('')
            readme_filename = None
            readme_content = ''
            readme_lang = 'en'
            for c in root_file_list:
                if not ('README' in c.name or 'readme' in c.name):
                    continue
                readme_filename = c.name
            if readme_filename is not None:
                repo_readme = repo.get_contents(readme_filename)
                if type(repo_readme) is list:
                    dl_url = repo_readme[0].download_url
                else:
                    dl_url = repo_readme.download_url
                if (dl_url is not None) and (dl_url != ''):
                    readme_content = requests.get(dl_url).text
                    try:
                        readme_lang = detect(readme_content[0:(512 if len(readme_content) <= 512 else -1)])
                    except LangDetectException:
                        readme_lang = 'en'
                if not readme_lang == 'en':
                    readme_content = self._chunk_translate_en(readme_content)
            hn_comments = ''
            while True:
                try:
                    hn_comments = self._get_hn_comments(repo.full_name)
                except Exception as e:
                    logging.warning(f"{e}")
                finally: 
                    break
            repo_object.link = repo.html_url
            repo_object.tags = repo.get_topics()
            repo_object.description = repo.description
            repo_object.orig_lang = readme_lang
            repo_object.readme = readme_content
            repo_object.hn_comments = hn_comments
            repo_object.updated = repo.updated_at.strftime('%Y/%m/%d, %H:%M:%S')
            repo_object.retrieved = datetime.now().strftime('%Y/%m/%d, %H:%M:%S')
    
    def _rebuild_rmap(self):
        for index, repo in enumerate(self.data):
            self.reverse_map[repo.name] = index
            
    def _append_rmap(
        self, 
        data: GitHubRepo
    ):
        self.reverse_map[data.name] = len(self.reverse_map)
    
    def _translate_wrapper(
        self, 
        x: str
    ):
        result = ''
        try:
            result = self._translator_artifact.translate(x) 
        except DTRequestError as e:
            if self._translate_err_counter <= self.TRANSLATE_MAX_RETRY:
                self._idle_handler.translate_rate_exceed_idle()
                self._translate_err_counter += 1
                result = self._translate_wrapper(x)
            else:
                x_list = x.split(' ')
                x_len = len(x_list) // 2
                magi_dataclasses_logger.warning(f'{e}, split into {x_len} and {len(x_list) - x_len}')
                result = self._translate_wrapper(' '.join(x_list[0:x_len])) + ' ' + self._translate_wrapper(' '.join(x_list[x_len::]))
        except ConnectionError as e:
            if self._translate_err_counter <= self.TRANSLATE_MAX_RETRY:
                self._translate_err_counter += 1
                result = self._translate_wrapper(x)
            else:
                result = ''
        self._translate_err_counter = 0
        return result
    
    def _divide(
        self,
        text: str, 
        chunk_len: int = 2048
    ) -> List[str]:
        n_chunks = len(text) // chunk_len
        return [
            text[i*chunk_len: i*chunk_len+chunk_len] if i != n_chunks - 1 else text[i*chunk_len::] for i in range(n_chunks)
        ]

    def _chunk_translate_en(
        self, 
        text: str
    ) -> str:
        if text is None:
            return ''
        if len(text) == 0:
            return ''
        try:
            return ''.join(
                list(
                    map(
                        self._translate_wrapper, self._divide(text)
                    )
                )
            )
        except TypeError:
            return ''
        
    def _get_hn_comments(
        self, 
        topic: str
    ) -> str:
        '''
        Arguments: 
            - topic (str) - form of f'{author_name}/{repo_name}' works best.
        Returns:
            - str - concatenated comments
        '''
        text = ''
        for index, r in enumerate(search_by_date(q=topic, stories=True, num_comments__gt=0)):
            if index >= 5:
                break
            hn_comments_raw = requests.get(f'http://hn.algolia.com/api/v1/items/{r["objectID"]}').json()['children']
            hn_comments_text = '<HN_SEP>'.join(
                [
                    BeautifulSoup(x['text'], features="lxml").text for x in hn_comments_raw if x['text'] is not None and len(x['text']) > 0
                ]
            )
            text += f"{hn_comments_text}<HN_SEP>"
        return text
    
    def _tokenize_corpus(
        self, 
        text: str, 
        n_process: int,
        _recursion_depth: int = 1
    ) -> List[str]:
        # the longest English word in Oxford dict is 45 characters long
        # 30 should be enough for most words
        corpus = [x for x in text.split(' ') if len(x) < 30]
        corpus_len = len(corpus)
        if len(text) > self._nlp_max_len:
            # if _resursion_depth is not too large,
            # split data in halves to achieve logarithm convergence
            # else iterate over the remainings
            if _recursion_depth <= 16:
                return self._tokenize_corpus(
                    text = ' '.join(corpus[0:corpus_len // 2]), 
                    n_process = n_process, 
                    _recursion_depth = _recursion_depth + 1
                ) + self._tokenize_corpus(
                    text = ' '.join(corpus[corpus_len // 2::]), 
                    n_process = n_process,
                    _recursion_depth = _recursion_depth + 1
                )
            else:
                chunk_num = len(text) // self._nlp_max_len + 1
                chunk_size = corpus_len // chunk_num + 1
                print(chunk_num, corpus_len, chunk_size)
                return list(
                    chain(
                        *[
                            self._tokenize_corpus(
                                text = ' '.join(corpus[i * chunk_size:(i + 1) * chunk_size]), 
                                n_process = n_process, 
                                _recursion_depth = _recursion_depth + 1
                            ) for i in range(chunk_num - 1)
                        ], 
                        self._tokenize_corpus(
                            text = ' '.join(corpus[(chunk_num - 1) * chunk_size::]), 
                            n_process = n_process, 
                            _recursion_depth = _recursion_depth + 1
                        )
                    )
                )
        else:
            doc = list(self._nlp_artifact.pipe(
                [text], 
                n_process = n_process
            ))[0]
            removal = ['ADJ' , 'ADV', 'PRON', 'CCONJ', 'PUNCT', 'PART', 'DET', 'ADP', 'SPACE']
            tokenized_corpus = [
                token.lemma_.lower() for token in doc \
                    if (not token.lemma_.lower() in self._nlp_artifact.Defaults.stop_words) and (token.is_alpha) and (len(token) > 2) and (token.pos_ not in removal) and token.lemma_.lower() not in self._stopwords
            ]
            return tokenized_corpus
    
    def _init_interlink_graph(self) -> None:
        self.G = nx.DiGraph()
        for index, repo in enumerate(self):
            self.G.add_node(
                index, 
                name = repo.name, 
                link = f'https://github.com/{repo.name}', 
                language = repo.lang, 
                topics = '',
                community = -1,
                stars = repo.stars
            )
        self._remove_code_chunk()
        self._remove_html_tags()
        for repo in self:
            for linked_repo in repo.mentioned_repos:
                if not linked_repo in self:
                    continue
                self.G.add_edge(
                    self.reverse_map[repo.name], 
                    self.reverse_map[linked_repo]
                )
        for key, func in self._graph_processors_artifacts.items():
            val_dict = func(G = self.G)
            nx.set_node_attributes(
                G = self.G, 
                values = val_dict, 
                name = key
            )
            magi_dataclasses_logger.info(
                f'Graph nodes attribute {key} calculated using {func}'
            )
        magi_dataclasses_logger.info(
            f'Graph built successfully.'
        )
    
    def remove_by_degree(
        self,
        min_remove_degree: int = 0
    ):
        assert self.G is not None, f'self.G is not initialized'
        remove = [
            node for node, degree in dict(self.G.degree()).items() if degree <= min_remove_degree
        ]
        self.G.remove_nodes_from(remove)
        magi_dataclasses_logger.info(
            f'Graph nodes with degree < {min_remove_degree}, total {len(remove)} nodes are removed.'
        )
    
    def init_repos(
        self, 
        fully_initialize: bool = False, 
        checkpoint_path: str = None
    ):
        control_idle = 0.1 if fully_initialize else 0.05
        self.data = []
        lang_report = defaultdict(int)
        for lang in self.LANG_LIST:
            magi_dataclasses_logger.info(f'Initializing language {lang}')
            repositories = self._github_artifact.search_repositories(query=f'stars:>{self.MIN_STAR_PER_REPO} language:{lang}')
            success = 0
            do_break = False
            for index, repo in tqdm(enumerate(repositories), total=self.MAX_REPO_PER_LANG):
                if do_break:
                    break
                while True:
                    self._idle_handler_artifact.github_rate_limit_control_idle(control_idle)
                    try:
                        if success >= self.MAX_REPO_PER_LANG:
                            do_break = True
                            break
                        repo_object = GitHubRepo(
                            name = repo.full_name,
                            lang = lang
                        )
                        if fully_initialize:
                            self._update_repo(repo_object, repo)
                        self.data.append(repo_object)
                        self.lang_stats[lang] += 1
                        success += 1
                        if success % self.CHECKPOINT_PERIOD == 0:
                            magi_dataclasses_logger.info(f'Coding language {lang}, initialization {success:5d} / {self.MAX_REPO_PER_LANG}')
                            if checkpoint_path:
                                self.dump(checkpoint_path)
                        break
                    except RateLimitExceededException as e:
                        self._idle_handler_artifact.github_rate_limit_exceed_idle()
            lang_report[lang] = index
        if checkpoint_path:
            self.dump(checkpoint_path)
        self._rebuild_rmap()
        for lang in self.LANG_LIST:
            magi_dataclasses_logger.info(f'Coding language {lang} retrieved with {index} repositories')

    def update_repos(self, checkpoint_path: str = None):
        for index in trange(len(self.data)):
            while True:
                self._idle_handler_artifact.github_rate_limit_control_idle(0.1)
                try:
                    self._update_repo(repo_object = self.data[index])
                    success += 1
                    if success % self.CHECKPOINT_PERIOD == 0:
                        magi_dataclasses_logger.info(f'Updated {success:5d} / {self.MAX_REPO_PER_LANG}')
                        if checkpoint_path:
                            self.dump(checkpoint_path)
                    break
                except RateLimitExceededException as e:
                    self._idle_handler_artifact.github_rate_limit_exceed_idle()
        if checkpoint_path:
            self.dump(checkpoint_path)
        magi_dataclasses_logger.info(f'Update complete, {success} repos updated.')       
        
    def load(self, file: Union[str, Path]):
        if type(file) is str:
            file = Path(file)
        file = file.resolve()
        assert file.exists(), f'{file} does not exist'
        assert file.suffix in ['.pkl', '.json'], f'Unsupported load type {file.suffix}'
        if file.suffix == '.pkl':
            with open(file, 'rb') as f:
                pickle_data_object = pickle.load(f)
            for d in pickle_data_object:
                assert type(d) is GitHubRepo, 'Pickled data must be of List[GitHubRepo] type'
            self.data = pickle_data_object
        elif file.suffix == '.json':
            with open(file ,'r') as f:
                json_data_object = json.load(f)
            assert type(json_data_object) is list, 'JSON data must be of List[dict] type'
            for index, d in enumerate(json_data_object):
                assert type(d) is dict, 'JSON data must be of List[dict] type'
                for k in GitHubRepo.__annotations__.keys():
                    if k[0] == '_': # reserved properties
                        continue
                    assert k in d.keys(), f'JSON data of index {index} missing key {k}'
            self.data = []
            for d in json_data_object:
                repo_object = GitHubRepo(
                    **{
                        k: d[k] for k in GitHubRepo.__annotations__.keys() if k[0] != '_'
                    },
                    _fully_initialized = True
                )
                self.data.append(repo_object)
        self.lang_stats = defaultdict(int)
        for d in self.data:
            self.lang_stats[d.lang] += 1
        self._rebuild_rmap()
        magi_dataclasses_logger.info(f'Loaded {len(self.data)} repos to {file}')
        
    def dump(self, file: Union[str, Path]):
        if type(file) is str:
            file = Path(file)
        file = file.resolve()
        assert file.suffix in ['.pkl', '.json'], f'Unsupported dump type {file.suffix}'
        if file.suffix == '.pkl':
            with open(file, 'wb') as f:
                pickle.dump(self.data, f)
        elif file.suffix == '.json':
             with open(file ,'w') as f:
                json_data_object = [
                    {
                        k: v for k, v in asdict(d).items() if k[0] != '_'
                    } for d in self.data
                ]
                json.dump(json_data_object, f)
        magi_dataclasses_logger.info(f'Dumped to {file}')
     
    def append(self, data):
        assert type(data) is GitHubRepo
        self.data.append(data)
        self.lang_stats[data.lang] += 1
        self._append_rmap(data)
        
    def filter_repos(self, patterns: Union[List[Union[str, re.Pattern]], str]):    
        if type(patterns) is str:
            with open(patterns, 'r') as f:
                patterns = [x.replace('\n', '') for x in f.readlines()]
        # patterns: List[Union[str, re.Pattern]]
        if type(patterns[0]) is str:
            patterns = [re.compile(r'{}'.format(p)) for p in patterns]
        new_data = []
        new_stats = defaultdict(int)
        filtered_stats = defaultdict(int)
        old_len = len(self.data)
        remove_list = []
        for repo in self.data:
            found = False
            for pattern in patterns:
                if len(pattern.findall(repo.name.split('/')[-1])) != 0:
                    found = True
                    filtered_stats[repo.lang] += 1
                    break
            if not found:
                new_data.append(repo)
                new_stats[repo.lang] += 1
            else:
                remove_list.append(repo.name)
        self.data = new_data
        self.lang_stats = new_stats
        
        magi_dataclasses_logger.info(f'{old_len - len(self.data)} repos filtered out using {len(patterns)} patterns.')
        magi_dataclasses_logger.info(f'filter_repos removal stats: {dict(filtered_stats)}')
        self._rebuild_rmap()
        return remove_list
        
    def get_corpus(self, idx: Union[str, int], n_process: int = 1):
        assert self._load_nlp, 'Calling get_corpus() on GitHubDataset object with load_nlp=False'
        text = self[idx].readme + self[idx].hn_comments
        self[idx].tokenized_corpus = tuple(
            self._tokenize_corpus(text, n_process)
        )
        return self[idx].tokenized_corpus
        
    @property   
    def statistics(self):
        gh_len = [len(x.readme.split(' ')) for x in self.data]
        hn_len = [len(x.hn_comments.split(' ')) for x in self.data]
        data_df = pd.DataFrame(
            data = [
                max(gh_len),
                min([x for x in gh_len if x > 0]),
                sum(gh_len) / len(gh_len),
                max(hn_len),
                min([x for x in hn_len if x > 0]),
                sum(hn_len) / len(hn_len),
                len([x for x in self.data if len(x.readme) > 0]) / len(self.data),
                len([x for x in self.data if len(x.hn_comments) > 0]) / len(self.data),
                max([x.stars for x in self.data]),
                min([x.stars for x in self.data]),
                sum([x.stars for x in self.data]) / len(self.data),
            ], 
            index = [
                'Max Length of README Corpus', 
                'Min Length of README Corpus',
                'Avg Length of README Corpus',
                'Max Length of HN Corpus',
                'Min Length of HN Corpus',
                'Avg Length of HN Corpus',
                '%Data with README file',
                '%Data with HN Comments',
                'Max Stars',
                'Min Stars', 
                'Avg Stars',
            ],
            columns = ['Value']
        )
        data_df.index.name = 'Statistics'
        return data_df
                
if __name__ == '__main__':
    # gd = GitHubDataset()
    # gd.init_repos(fully_initialize=True, checkpoint_path='ghv9.json')
    github_dataset = GitHubDataset(
        empty = False,
        file_path = './ghv9-3.json',
        load_nlp = True,
        load_graph = True
    )
    
    