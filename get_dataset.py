def get_dataset(dataset: str, args):
    if dataset == 'arxiv':
        if args.method in ['coral', 'groupdro', 'irm']:
            from data.arxiv.data_generator import ArXivGroup
            return ArXivGroup(args)
        else:
            from data.arxiv.data_generator import ArXiv
            return ArXiv(args)

    elif dataset == 'drug':
        if args.method in ['coral', 'groupdro', 'irm']:
            from data.drug.data_generator import TdcDtiDgGroup
            return TdcDtiDgGroup(args)
        else:
            from data.drug.data_generator import TdcDtiDg
            return TdcDtiDg(args)

    elif dataset == 'fmow':
        if args.method in ['coral', 'groupdro', 'irm']:
            from data.fmow.data_generator import FMoWGroup
            return FMoWGroup(args)
        else:
            from data.fmow.data_generator import FMoW
            return FMoW(args)

    elif dataset == 'huffpost':
        if args.method in ['coral', 'groupdro', 'irm']:
            from data.huffpost.data_generator import HuffPostGroup
            return HuffPostGroup(args)
        else:
            from data.huffpost.data_generator import HuffPost
            return HuffPost(args)

    elif dataset == 'mimic':
        if args.method in ['coral', 'groupdro', 'irm']:
            from data.MIMIC.data_generator import MIMICGroup
            return MIMICGroup(args)
        else:
            from data.MIMIC.data_generator import MIMIC
            return MIMIC(args)

    elif dataset == 'precipitation':
        if args.method in ['coral', 'groupdro', 'irm']:
            from data.precipitation.data_generator import PrecipitationGroup
            return PrecipitationGroup(args)
        else:
            from data.precipitation.data_generator import Precipitation
            return Precipitation(args)

    elif dataset == 'yearbook':
        if args.method in ['coral', 'groupdro', 'irm']:
            from data.yearbook.data_generator import YearbookGroup
            return YearbookGroup(args)
        else:
            from data.yearbook.data_generator import Yearbook
            return Yearbook(args)

