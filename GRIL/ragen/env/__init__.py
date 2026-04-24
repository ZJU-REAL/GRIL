# from .alfworld.config import AlfredEnvConfig
# from .alfworld.env import AlfredTXTEnv
from .bandit.config import BanditEnvConfig
from .bandit.env import BanditEnv
from .countdown.config import CountdownEnvConfig
from .countdown.env import CountdownEnv
from .sokoban.config import SokobanEnvConfig
from .sokoban.env import SokobanEnv
from .frozen_lake.config import FrozenLakeEnvConfig
from .frozen_lake.env import FrozenLakeEnv
from .metamathqa.env import MetaMathQAEnv
from .metamathqa.config import MetaMathQAEnvConfig
from .premise_detect.env import PremiseDetectEnv
from .premise_detect.config import PremiseDetectEnvConfig
from .premise_evaluate.env import PremiseEvaluateEnv
from .premise_evaluate.config import PremiseEvaluateEnvConfig
from .hotpot_insufficient.env import HotpotQAInsufficientEnv
from .hotpot_insufficient.config import HotpotQAInsufficientEnvConfig
from .hotpot_full.env import HotpotQAEnv
from .hotpot_full.config import HotpotQAEnvConfig

REGISTERED_ENVS = {
    'bandit': BanditEnv,
    'countdown': CountdownEnv,
    'sokoban': SokobanEnv,
    'frozen_lake': FrozenLakeEnv,
    # 'alfworld': AlfredTXTEnv,
    'metamathqa': MetaMathQAEnv,
    'premise': PremiseDetectEnv,
    'premise_evaluate': PremiseEvaluateEnv,
    'hotpot_insufficient': HotpotQAInsufficientEnv,
    'hotpot_full': HotpotQAEnv,
}

REGISTERED_ENV_CONFIGS = {
    'bandit': BanditEnvConfig,
    'countdown': CountdownEnvConfig,
    'sokoban': SokobanEnvConfig,
    'frozen_lake': FrozenLakeEnvConfig,
    # 'alfworld': AlfredEnvConfig,
    'metamathqa': MetaMathQAEnvConfig,
    'premise': PremiseDetectEnvConfig,
    'premise_evaluate': PremiseEvaluateEnvConfig,
    'hotpot_insufficient': HotpotQAInsufficientEnvConfig,
    'hotpot_full': HotpotQAEnvConfig,
}

try:
    from .webshop.env import WebShopEnv
    from .webshop.config import WebShopEnvConfig
    REGISTERED_ENVS['webshop'] = WebShopEnv
    REGISTERED_ENV_CONFIGS['webshop'] = WebShopEnvConfig
except ImportError:
    pass
