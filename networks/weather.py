import ipdb
import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from pytorch_tabular.models.base_model import BaseModel
from pytorch_tabular.config import ModelConfig, DataConfig
# from pytorch_tabular.models.ft_transformer.ft_transformer import FTTransformerModel, FTTransformerBackbone
from networks.fttransformer import FTTransformerBackbone
from pytorch_tabular.models.ft_transformer.config import FTTransformerConfig

continuous_cols = [
    "climate_pressure", "climate_temperature", "cmc_0_0_0_1000", "cmc_0_0_0_2", "cmc_0_0_0_2_grad",
    "cmc_0_0_0_2_interpolated", "cmc_0_0_0_2_next", "cmc_0_0_0_500", "cmc_0_0_0_700", "cmc_0_0_0_850",
    "cmc_0_0_0_925", "cmc_0_0_6_2", "cmc_0_0_7_1000", "cmc_0_0_7_2", "cmc_0_0_7_500", "cmc_0_0_7_700",
    "cmc_0_0_7_850", "cmc_0_0_7_925", "cmc_0_1_0_0", "cmc_0_1_11_0", "cmc_0_1_65_0", "cmc_0_1_65_0_grad",
    "cmc_0_1_65_0_next", "cmc_0_1_66_0", "cmc_0_1_66_0_grad", "cmc_0_1_66_0_next", "cmc_0_1_67_0",
    "cmc_0_1_67_0_grad", "cmc_0_1_67_0_next", "cmc_0_1_68_0", "cmc_0_1_68_0_grad", "cmc_0_1_68_0_next",
    "cmc_0_1_7_0", "cmc_0_2_2_10", "cmc_0_2_2_1000", "cmc_0_2_2_500", "cmc_0_2_2_700", "cmc_0_2_2_850",
    "cmc_0_2_2_925", "cmc_0_2_3_10", "cmc_0_2_3_1000", "cmc_0_2_3_500", "cmc_0_2_3_700", "cmc_0_2_3_850",
    "cmc_0_2_3_925", "cmc_0_3_0_0", "cmc_0_3_0_0_next", "cmc_0_3_1_0", "cmc_0_3_5_1000", "cmc_0_3_5_500",
    "cmc_0_3_5_700", "cmc_0_3_5_850", "cmc_0_3_5_925", "cmc_0_6_1_0", "cmc_available", "cmc_horizon_h",
    "cmc_precipitations", "cmc_timedelta_s", "gfs_2m_dewpoint", "gfs_2m_dewpoint_grad", "gfs_2m_dewpoint_next",
    "gfs_a_vorticity", "gfs_available", "gfs_cloudness", "gfs_clouds_sea", "gfs_horizon_h", "gfs_humidity",
    "gfs_precipitable_water", "gfs_precipitations", "gfs_pressure", "gfs_r_velocity", "gfs_soil_temperature",
    "gfs_soil_temperature_available", "gfs_temperature_10000", "gfs_temperature_15000", "gfs_temperature_20000",
    "gfs_temperature_25000", "gfs_temperature_30000", "gfs_temperature_35000", "gfs_temperature_40000",
    "gfs_temperature_45000", "gfs_temperature_5000", "gfs_temperature_50000", "gfs_temperature_55000", "gfs_temperature_60000",
    "gfs_temperature_65000", "gfs_temperature_7000", "gfs_temperature_70000", "gfs_temperature_75000", "gfs_temperature_80000",
    "gfs_temperature_85000", "gfs_temperature_90000", "gfs_temperature_92500", "gfs_temperature_95000", "gfs_temperature_97500",
    "gfs_temperature_sea", "gfs_temperature_sea_grad", "gfs_temperature_sea_interpolated", "gfs_temperature_sea_next",
    "gfs_timedelta_s", "gfs_total_clouds_cover_high", "gfs_total_clouds_cover_low", "gfs_total_clouds_cover_low_grad",
    "gfs_total_clouds_cover_low_next", "gfs_total_clouds_cover_middle", "gfs_u_wind", "gfs_v_wind", "gfs_wind_speed",
    "sun_elevation", "topography_bathymetry", "wrf_graupel", "wrf_hail", "wrf_psfc", "wrf_rain", "wrf_rh2", "wrf_snow",
    "wrf_t2", "wrf_t2_grad", "wrf_t2_interpolated", "wrf_t2_next", "wrf_wind_u", "wrf_wind_v"
]

categorical_cols = [
    "wrf_available"
]

def read_parse_config(config, cls):
    if isinstance(config, str):
        if os.path.exists(config):
            _config = OmegaConf.load(config)
            if cls == ModelConfig:
                cls = getattr(
                    getattr(models, _config._module_src), _config._config_name
                )
            config = cls(
                **{
                    k: v
                    for k, v in _config.items()
                    if (k in cls.__dataclass_fields__.keys())
                       and (cls.__dataclass_fields__[k].init)
                }
            )
        else:
            raise ValueError(f"{config} is not a valid path")
    config = OmegaConf.structured(config)
    return config


class WeatherNetwork(pl.LightningModule):
    def __init__(self, args, num_classes):
        super(WeatherNetwork, self).__init__()
        self.args = args
        self.num_classes = num_classes
        task = "regression" if self.args.regression else "classification"
        lr = self.args.lr

        if task == "regression":
            metrics = ["f1", "accuracy"]
            metrics_params = [{"num_classes": self.num_classes, "average": "macro"}, {}]
            target_name = ["fact_temperature"]
        else:
            metrics = ["mae", "rmse"]
            metrics_params = [{}, {}]
            target_name = ["fact_cwsm_class"]
        model_config = FTTransformerConfig(
            task=task,
            embedding_dims=[],
            learning_rate=lr,
            out_ff_layers="32-32",
            out_ff_activation="LeakyReLU",
            # metrics=metrics,
            # metrics_params=metrics_params,
            target_range=None
        )
        data_config = DataConfig(
            target=target_name,
            date_columns=[("time", "M")],
            continuous_cols=continuous_cols,
            categorical_cols=categorical_cols
        )
        self.config = OmegaConf.merge(
            OmegaConf.to_container(read_parse_config(model_config, ModelConfig)),
            OmegaConf.to_container(read_parse_config(data_config, DataConfig))
        )

        # Set to default values, as in
        # https://github.com/manujosephv/pytorch_tabular/blob/76e83fb396d02a7b883e03f8c2ea8556ff472798/pytorch_tabular/models/ft_transformer/config.py#L12
        # self.config = OmegaConf.create({
        #     "share_embedding": True,
        #     "embedding_bias": True,
        #     "share_embedding_strategy": "fraction",
        #     "shared_embedding_fraction": 0.25,
        #     "input_embed_dim": 32,
        #     "embedding_initialization": None,
        #     "num_attn_blocks": 3,
        #     "num_heads": 6,
        #     "ff_hidden_multiplier": 4,
        #     "transformer_activation": "GEGLU",
        #     "attn_dropout": 0.1,
        #     "keep_attn": True,
        #     "ff_dropout": 0.1,
        #     "add_norm_dropout": 0.1,
        #     "attn_feature_importance": True,
        #     "batch_norm_continuous_input": True,
        #     "transformer_head_dim": None,
        #     "out_ff_layers": "1024-256",
        #     "out_ff_initialization": "kaiming",
        #     "out_ff_activation": "ReLU",
        #     "embedding_dropout": 0,
        #     "out_ff_dropout": 0.1,
        #     "use_batch_norm": True,
        #     "categorical_dim": 123,
        #     "categorical_cols": [],
        #     "continuous_dim": 1,
        #     "continuous_cols": []})
        self.net = FTTransformerBackbone(self.config).cuda()
        self.logits = nn.Linear(32, self.num_classes).cuda()

    def apply_output_sigmoid_scaling(self, y_hat: torch.Tensor):
        if (self.hparams.task == "regression") and (
            self.hparams.target_range is not None
        ):
            for i in range(self.hparams.output_dim):
                y_min, y_max = self.hparams.target_range[i]
                y_hat[:, i] = y_min + nn.Sigmoid()(y_hat[:, i]) * (y_max - y_min)
        return y_hat

    def forward(self, x):
        backbone_features = self.net.forward(x)
        logits = self.net.linear_layers(backbone_features)
        logits = self.logits(logits)
        # y_hat = self.apply_output_sigmoid_scaling(logits)
        return logits



# class WeatherNetwork(nn.Module):
#     def __init__(self, args, num_input_channels, num_classes):
#         super(WeatherNetwork, self).__init__()
#         self.args = args
#         self.net = nn.Sequential(self.fc_block(num_input_channels, 128), self.fc_block(128, 64))
#         self.logits = nn.Linear(64, num_classes)
#         self.num_classes = num_classes
#         self.leaky_relu = torch.nn.LeakyReLU()
#
#     def fc_block(self, in_channels, out_channels):
#         return nn.Sequential(
#             nn.Linear(in_channels, out_channels),
#             nn.BatchNorm1d(out_channels),
#             nn.LeakyReLU()
#         )
#
#     def forward(self, x):
#         x = self.net(x)
#
#         x = x.view(x.size(0), -1)
#
#         if self.num_classes == 1:
#             return self.leaky_relu(self.logits(x))
#         else:
#             return self.logits(x)
