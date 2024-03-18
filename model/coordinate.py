from astropy import coordinates as coord
import astropy.units as u
from astropy.coordinates import SkyCoord
import numpy as np


def transition(ra_list, dec_list, ra, dec):
    try:
        ra_all = [ra11.text.replace("\n", "") for ra11 in ra_list]
        dec_all = [dec11.text.replace("\n", "") for dec11 in dec_list]
        coords = [SkyCoord(ra=ra11, dec=dec11, unit=(u.hourangle, u.deg)) for ra11, dec11 in zip(ra_all, dec_all)]
        ras = [coord.ra.deg for coord in coords]
        decs = [coord.dec.deg for coord in coords]
        distances = np.sqrt((np.array(ras) - ra) ** 2 + (np.array(decs) - dec) ** 2)

        # 筛选距离小于0.5的值的索引
        close_indices = np.where(distances < 0.01)[0]

        # 找到距离数组中最小值的索引
        if len(close_indices) > 0:
            # 找到距离数组中最小值的索引
            closest_index = close_indices[np.argmin(distances[close_indices])]

            return ras, decs, closest_index
        else:
            # 没有满足条件的值，返回空值
            return None,None,None

    except:
        return 0, 0, 0


def clean_titles(titles):
    # 去除每个标题中的换行符，并过滤掉只包含空白字符的元素
    if titles==None:
        return 0
    else:
        cleaned_titles = [title.replace('\n', '').strip() for title in titles if title.strip()]
        return cleaned_titles





