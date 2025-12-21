import logging
from typing import Tuple

import numpy as np
import pandas as pd

from market_monitor.utils.enums import ISIN_TO_TICKER

logger = logging.getLogger()


class StockSelector:

    def __init__(self, weight_matrix: pd.DataFrame, max_n_of_stocks: int = 2500):

        self.weight_matrix: pd.DataFrame = weight_matrix
        self.isin_equiduct: list[str] = [] #ISINS_EQUIDUCT
        self.max_n_of_stocks: int = max_n_of_stocks
        self.compress_stocks()

    def compress_stocks(self):

        etf_to_drop = self.check_for_stock_in_single_etf()
        self.drop_etf(etf_to_drop)
        self.check_for_negligible_stocks()

    def truncate_stocks(self, cutoff: int | None = None) -> Tuple[list, list]:

        stock_sorted_by_weight = self.weight_matrix.loc[:,
                                 self.weight_matrix.sum().abs().sort_values(ascending=False).index]
        bbg_added_isins = [isin for isin in stock_sorted_by_weight.columns if isin not in self.isin_equiduct]
        num_all_stocks = len(bbg_added_isins)
        n_th_isin = min(cutoff, num_all_stocks) - 1 if cutoff is not None else num_all_stocks
        n_th_isin_index = stock_sorted_by_weight.columns.get_loc(bbg_added_isins[n_th_isin])
        stock_to_keep = stock_sorted_by_weight.columns[:n_th_isin_index].to_list()
        return stock_to_keep, self.weight_matrix.index.tolist()

    def check_for_stock_in_single_etf(self, threshold=500):
        is_present_matrix = self.weight_matrix.drop(self.isin_equiduct, errors="ignore", axis=1).astype(bool)
        stock_just_in_1_etf = is_present_matrix.columns[is_present_matrix.sum() < 2]
        etf_that_adds_stocks = is_present_matrix[stock_just_in_1_etf] \
            .sum(axis=1) \
            .sort_values(ascending=False)
        etfs_to_drop = etf_that_adds_stocks[etf_that_adds_stocks > threshold]
        if not etfs_to_drop.empty: print(etfs_to_drop.rename(ISIN_TO_TICKER).to_string())
        return etfs_to_drop.index.tolist()

    def check_stock_by_weight(self, cutoff, threshold):

        stock_sorted_by_weight = self.weight_matrix.loc[:,
                                 self.weight_matrix.sum().abs().sort_values(ascending=False).index]
        added_isins = [isin for isin in stock_sorted_by_weight.columns if isin not in self.isin_equiduct]
        n_th_isin = added_isins[min(cutoff, len(added_isins))]
        n_th_isin_index = stock_sorted_by_weight.columns.get_loc(n_th_isin)
        stock_truncated = stock_sorted_by_weight.iloc[:, :n_th_isin_index]
        n_stocks = stock_truncated.sum(axis=1) > threshold
        return np.mean(n_stocks)

    def drop_etf(self, etfs):
        logger.info("deleting etfs that adds to many stocks")
        self.weight_matrix.drop(etfs, inplace=True)
        self.weight_matrix = self.weight_matrix.loc[:, self.weight_matrix.astype(bool).any()]
        logger.info(f"... n_stocks now is {len(self.weight_matrix.columns)}")

    def check_for_negligible_stocks(self):
        stock_ranking = self.weight_matrix.drop(self.isin_equiduct,
                                                axis=1,
                                                errors='ignore').sum().sort_values()

        self.stock_to_drop = stock_ranking[stock_ranking.abs() < 0.0005].index
        if len(self.stock_to_drop):
            logger.info(f"...dropping {len(self.stock_to_drop)} stocks...")
            self.weight_matrix = self.weight_matrix.drop(self.stock_to_drop, axis=1)
            logger.info(f"...n_stock is now:"
                  f" {len(self.weight_matrix.drop(self.isin_equiduct, axis=1, errors='ignore').columns)}")
        else:
            print("...no negligible stock found...")

    def get_stock_to_drop(self):
        return self.stock_to_drop


if __name__ == "__main__":
    isins = ["IE00BPVLQD13",
             "LU0950674175",
             "LU1291097779",
             "IE00BTJRMP35",
             "LU0514695690",
             "IE00B95PGT31",
             "IE00BK5BQW10",
             "FR0010315770",
             "LU1900067940",
             "LU0480132876",
             "LU1781541849",
             "IE00BLRPN388",
             "LU2573967036",
             "IE00B5L8K969",
             "LU2573966905",
             "IE00B469F816",
             "FR0010261198",
             "IE00B4L5YC18",
             "IE00BKWQ0Q14",
             "IE00BDGN9Z19",
             "IE00B910VR50",
             "LU1291098827",
             "LU1931974262",
             "IE00B945VV12",
             "IE00BKM4GZ66",
             "IE00B53QG562",
             "IE00BHZPJ239",
             "FR0010429068",
             "IE00B53L3W79",
             "LU2109787049",
             "LU0274211217",
             "IE00B466KX20",
             "IE00B4L5YX21",
             "LU1681044480",
             "IE00BJ0KDR00",
             "IE00BFNM3J75",
             "IE00B3ZW0K18",
             "LU1931974429",
             "LU1781541179",
             "IE00BJ0KDQ92",
             "IE00B4L5Y983",
             "IE00B5BMR087",
             "FR0014003IY1",
             "IE000QDFFK00",
             "LU0328475792",
             "IE00B52SFT06",
             "IE00BHZPJ908",
             "IE00BFMXXD54",
             "IE000G2LIHG9",
             "IE00B53SZB19",
             "IE00B02KXH56",
             "IE00BD4TY451",
             "LU1681038672",
             "IE0031442068",
             "LU1681042609",
             "LU0496786574",
             "IE00BK5BQV03",
             "IE00B53QDK08",
             "LU0959211326",
             "IE00BFXR5T61",
             "IE00B3CNHJ55",
             "LU0136240974",
             "LU1681043599",
             "LU1681038243",
             "LU0908500753",
             "LU1841731745",
             "IE00BD4TYG73",
             "IE00B60SX394",
             "LU1781541252",
             "IE00BFZXGZ54",
             "LU0136234654",
             "FR0010245514",
             "IE00BHZPJ569",
             "IE00BKX55T58",
             "LU0950671825",
             "IE00B77D4428",
             "IE00BM67HW99",
             "FR0011550185",
             "IE00BFY0GT14",
             "IE0032077012",
             "LU0959211243",
             "IE00BJ38QD84",
             "LU0340285161",
             "LU1681049109",
             "LU1681048804",
             "IE00B3XXRP09",
             "LU1900068914",
             "LU2456436083",
             "IE00B02KXK85",
             "IE00BMFKG444",
             "IE00BCHWNQ94",
             "LU1781540957",
             "LU2314312849",
             "LU0292109856",
             "IE00BJZ2DC62",
             "IE00BFXR5Q31",
             "IE00B7K93397",
             "IE000TSML5I8",
             "IE000Z9SJA06",
             "LU0328474803",
             "LU1953188833",
             "IE00BD4TXS21",
             "LU0274209740",
             "LU1931974775",
             "IE00BHZRR147",
             "LU0380865021",
             "IE00BKLTRN76",
             "FR0007054358",
             "IE00BYYW2V44",
             "LU1681047236",
             "IE00B0M63177",
             "IE0001JH5CB4",
             "LU0147308422",
             "LU0446734104",
             "IE00BCLWRF22",
             "LU1600334798",
             "IE00BKSBGT50",
             "LU0136234068",
             "IE00BK5BQX27",
             "IE00BHZPJ783",
             "IE00BYXZ2585",
             "LU1900068161",
             "LU0322253732",
             "LU1437015735",
             "LU0846194776",
             "LU0274209237",
             "IE00BJQRDL90",
             "IE00B4K48X80",
             "LU0950668870",

             ]


