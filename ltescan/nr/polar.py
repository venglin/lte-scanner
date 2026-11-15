"""Polar code of TS 38.212 5.3.1 / 5.4.1 (downlink, without parity-check
bits, as used by the PBCH and the DCI): input bit interleaver, code length
selection, frozen set from the reliability sequence (with the extra frozen
bits of puncturing and shortening), encoder, sub-block interleaving rate
matching (repetition, puncturing, shortening) and a CRC-aided
successive-cancellation list decoder.

LLR convention as elsewhere: positive means bit 0."""
from __future__ import annotations

from functools import lru_cache

import numpy as np

# 38.212 Table 5.3.1.2-1: polar reliability sequence Q_0^{1023}, in
# ascending order of reliability
Q_MAX = (
    0, 1, 2, 4, 8, 16, 32, 3, 5, 64, 9, 6, 17, 10, 18, 128, 12, 33, 65, 20, 256, 34, 24, 36,
    7, 129, 66, 512, 11, 40, 68, 130, 19, 13, 48, 14, 72, 257, 21, 132, 35, 258, 26, 513, 80,
    37, 25, 22, 136, 260, 264, 38, 514, 96, 67, 41, 144, 28, 69, 42, 516, 49, 74, 272, 160,
    520, 288, 528, 192, 544, 70, 44, 131, 81, 50, 73, 15, 320, 133, 52, 23, 134, 384, 76, 137,
    82, 56, 27, 97, 39, 259, 84, 138, 145, 261, 29, 43, 98, 515, 88, 140, 30, 146, 71, 262,
    265, 161, 576, 45, 100, 640, 51, 148, 46, 75, 266, 273, 517, 104, 162, 53, 193, 152, 77,
    164, 768, 268, 274, 518, 54, 83, 57, 521, 112, 135, 78, 289, 194, 85, 276, 522, 58, 168,
    139, 99, 86, 60, 280, 89, 290, 529, 524, 196, 141, 101, 147, 176, 142, 530, 321, 31, 200,
    90, 545, 292, 322, 532, 263, 149, 102, 105, 304, 296, 163, 92, 47, 267, 385, 546, 324, 208,
    386, 150, 153, 165, 106, 55, 328, 536, 577, 548, 113, 154, 79, 269, 108, 578, 224, 166,
    519, 552, 195, 270, 641, 523, 275, 580, 291, 59, 169, 560, 114, 277, 156, 87, 197, 116,
    170, 61, 531, 525, 642, 281, 278, 526, 177, 293, 388, 91, 584, 769, 198, 172, 120, 201,
    336, 62, 282, 143, 103, 178, 294, 93, 644, 202, 592, 323, 392, 297, 770, 107, 180, 151,
    209, 284, 648, 94, 204, 298, 400, 608, 352, 325, 533, 155, 210, 305, 547, 300, 109, 184,
    534, 537, 115, 167, 225, 326, 306, 772, 157, 656, 329, 110, 117, 212, 171, 776, 330, 226,
    549, 538, 387, 308, 216, 416, 271, 279, 158, 337, 550, 672, 118, 332, 579, 540, 389, 173,
    121, 553, 199, 784, 179, 228, 338, 312, 704, 390, 174, 554, 581, 393, 283, 122, 448, 353,
    561, 203, 63, 340, 394, 527, 582, 556, 181, 295, 285, 232, 124, 205, 182, 643, 562, 286,
    585, 299, 354, 211, 401, 185, 396, 344, 586, 645, 593, 535, 240, 206, 95, 327, 564, 800,
    402, 356, 307, 301, 417, 213, 568, 832, 588, 186, 646, 404, 227, 896, 594, 418, 302, 649,
    771, 360, 539, 111, 331, 214, 309, 188, 449, 217, 408, 609, 596, 551, 650, 229, 159, 420,
    310, 541, 773, 610, 657, 333, 119, 600, 339, 218, 368, 652, 230, 391, 313, 450, 542, 334,
    233, 555, 774, 175, 123, 658, 612, 341, 777, 220, 314, 424, 395, 673, 583, 355, 287, 183,
    234, 125, 557, 660, 616, 342, 316, 241, 778, 563, 345, 452, 397, 403, 207, 674, 558, 785,
    432, 357, 187, 236, 664, 624, 587, 780, 705, 126, 242, 565, 398, 346, 456, 358, 405, 303,
    569, 244, 595, 189, 566, 676, 361, 706, 589, 215, 786, 647, 348, 419, 406, 464, 680, 801,
    362, 590, 409, 570, 788, 597, 572, 219, 311, 708, 598, 601, 651, 421, 792, 802, 611, 602,
    410, 231, 688, 653, 248, 369, 190, 364, 654, 659, 335, 480, 315, 221, 370, 613, 422, 425,
    451, 614, 543, 235, 412, 343, 372, 775, 317, 222, 426, 453, 237, 559, 833, 804, 712, 834,
    661, 808, 779, 617, 604, 433, 720, 816, 836, 347, 897, 243, 662, 454, 318, 675, 618, 898,
    781, 376, 428, 665, 736, 567, 840, 625, 238, 359, 457, 399, 787, 591, 678, 434, 677, 349,
    245, 458, 666, 620, 363, 127, 191, 782, 407, 436, 626, 571, 465, 681, 246, 707, 350, 599,
    668, 790, 460, 249, 682, 573, 411, 803, 789, 709, 365, 440, 628, 689, 374, 423, 466, 793,
    250, 371, 481, 574, 413, 603, 366, 468, 655, 900, 805, 615, 684, 710, 429, 794, 252, 373,
    605, 848, 690, 713, 632, 482, 806, 427, 904, 414, 223, 663, 692, 835, 619, 472, 455, 796,
    809, 714, 721, 837, 716, 864, 810, 606, 912, 722, 696, 377, 435, 817, 319, 621, 812, 484,
    430, 838, 667, 488, 239, 378, 459, 622, 627, 437, 380, 818, 461, 496, 669, 679, 724, 841,
    629, 351, 467, 438, 737, 251, 462, 442, 441, 469, 247, 683, 842, 738, 899, 670, 783, 849,
    820, 728, 928, 791, 367, 901, 630, 685, 844, 633, 711, 253, 691, 824, 902, 686, 740, 850,
    375, 444, 470, 483, 415, 485, 905, 795, 473, 634, 744, 852, 960, 865, 693, 797, 906, 715,
    807, 474, 636, 694, 254, 717, 575, 913, 798, 811, 379, 697, 431, 607, 489, 866, 723, 486,
    908, 718, 813, 476, 856, 839, 725, 698, 914, 752, 868, 819, 814, 439, 929, 490, 623, 671,
    739, 916, 463, 843, 381, 497, 930, 821, 726, 961, 872, 492, 631, 729, 700, 443, 741, 845,
    920, 382, 822, 851, 730, 498, 880, 742, 445, 471, 635, 932, 687, 903, 825, 500, 846, 745,
    826, 732, 446, 962, 936, 475, 853, 867, 637, 907, 487, 695, 746, 828, 753, 854, 857, 504,
    799, 255, 964, 909, 719, 477, 915, 638, 748, 944, 869, 491, 699, 754, 858, 478, 968, 383,
    910, 815, 976, 870, 917, 727, 493, 873, 701, 931, 756, 860, 499, 731, 823, 922, 874, 918,
    502, 933, 743, 760, 881, 494, 702, 921, 501, 876, 847, 992, 447, 733, 827, 934, 882, 937,
    963, 747, 505, 855, 924, 734, 829, 965, 938, 884, 506, 749, 945, 966, 755, 859, 940, 830,
    911, 871, 639, 888, 479, 946, 750, 969, 508, 861, 757, 970, 919, 875, 862, 758, 948, 977,
    923, 972, 761, 877, 952, 495, 703, 935, 978, 883, 762, 503, 925, 878, 735, 993, 885, 939,
    994, 980, 926, 764, 941, 967, 886, 831, 947, 507, 889, 984, 751, 942, 996, 971, 890, 509,
    949, 973, 1000, 892, 950, 863, 759, 1008, 510, 979, 953, 763, 974, 954, 879, 981, 982, 927,
    995, 765, 956, 887, 985, 997, 986, 943, 891, 998, 766, 511, 988, 1001, 951, 1002, 893, 975,
    894, 1009, 955, 1004, 1010, 957, 983, 958, 987, 1012, 999, 1016, 767, 989, 1003, 990, 1005,
    959, 1011, 1013, 895, 1006, 1014, 1017, 1018, 991, 1020, 1007, 1015, 1019, 1021, 1022, 1023,
)

# 38.212 Table 5.3.1.1-1: interleaving pattern PI_IL^max (K_IL^max = 164)
PI_IL_MAX = (
    0, 2, 4, 7, 9, 14, 19, 20, 24, 25, 26, 28, 31, 34, 42, 45, 49, 50, 51, 53, 54, 56, 58, 59,
    61, 62, 65, 66, 67, 69, 70, 71, 72, 76, 77, 81, 82, 83, 87, 88, 89, 91, 93, 95, 98, 101,
    104, 106, 108, 110, 111, 113, 115, 118, 119, 120, 122, 123, 126, 127, 129, 132, 134, 138,
    139, 140, 1, 3, 5, 8, 10, 15, 21, 27, 29, 32, 35, 43, 46, 52, 55, 57, 60, 63, 68, 73, 78,
    84, 90, 92, 94, 96, 99, 102, 105, 107, 109, 112, 114, 116, 121, 124, 128, 130, 133, 135,
    141, 6, 11, 16, 22, 30, 33, 36, 44, 47, 64, 74, 79, 85, 97, 100, 103, 117, 125, 131, 136,
    142, 12, 17, 23, 37, 48, 75, 80, 86, 137, 143, 13, 18, 38, 144, 39, 145, 40, 146, 41, 147,
    148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163,
)

# 38.212 Table 5.4.1.1-1: sub-block interleaver pattern P(i)
SUBBLOCK_P = (0, 1, 2, 4, 3, 5, 6, 7, 8, 16, 9, 17, 10, 18, 11, 19, 12, 20, 13, 21, 14, 22, 15,
              23, 24, 25, 26, 28, 27, 29, 30, 31)


@lru_cache(maxsize=16)
def input_interleaver(k: int) -> np.ndarray:
    """PI(k) of 38.212 5.3.1.1 (I_IL = 1): c'_k = c_PI(k)."""
    kmax = len(PI_IL_MAX)
    if k > kmax:
        raise ValueError("K too large for the input interleaver")
    p = np.array([v - (kmax - k) for v in PI_IL_MAX if v >= kmax - k], dtype=np.int64)
    p.setflags(write=False)
    return p


@lru_cache(maxsize=16)
def info_set(k: int, n: int) -> np.ndarray:
    """Sorted indices of the K information bits for code length N (no
    puncturing/shortening, i.e. E >= N, no parity-check bits)."""
    q = [v for v in Q_MAX if v < n]
    s = np.sort(np.array(q[-k:], dtype=np.int64))
    s.setflags(write=False)
    return s


def code_length(k: int, e: int, n_max: int = 9) -> int:
    """N of 38.212 5.3.1 (R_min = 1/8, n_min = 5)."""
    ce = int(np.ceil(np.log2(e)))
    n1 = ce - 1 if (8 * e <= 9 * 2 ** (ce - 1) and 16 * k < 9 * e) else ce
    n2 = int(np.ceil(np.log2(8 * k)))
    return 2 ** max(min(n1, n2, n_max), 5)


def _punctured(k: int, e: int) -> bool:
    return 16 * k <= 7 * e


@lru_cache(maxsize=64)
def info_set_rm(k: int, n: int, e: int) -> np.ndarray:
    """Information bit indices for rate-matched length E (38.212 5.3.1.2:
    with E < N the punctured or shortened positions, and for puncturing
    also the lowest indices, are frozen first)."""
    if e >= n:
        return info_set(k, n)
    j = subblock_index(n)
    u = n - e
    if _punctured(k, e):
        frz = set(int(v) for v in j[:u])
        t = int(np.ceil(3 * n / 4 - e / 2)) if e >= 3 * n / 4 else int(np.ceil(9 * n / 16 - e / 4))
        frz |= set(range(t))
    else:
        frz = set(int(v) for v in j[e:])
    q = [v for v in Q_MAX if v < n and v not in frz]
    s = np.sort(np.array(q[-k:], dtype=np.int64))
    s.setflags(write=False)
    return s


@lru_cache(maxsize=16)
def subblock_index(n: int) -> np.ndarray:
    """J(n) of 38.212 5.4.1.1: y_n = d_J(n)."""
    i = np.arange(n)
    j = np.array(SUBBLOCK_P)[(32 * i) // n] * (n // 32) + i % (n // 32)
    j.setflags(write=False)
    return j


def polar_transform(u: np.ndarray) -> np.ndarray:
    """x = u G_N (mod 2), G_N = n-fold Kronecker power of [[1,0],[1,1]]."""
    x = np.array(u, dtype=np.uint8)
    n = len(x)
    s = 1
    while s < n:
        x = x.reshape(-1, 2 * s)
        x[:, :s] ^= x[:, s:]
        x = x.reshape(n)
        s *= 2
    return x


def encode(c: np.ndarray, n: int, e: int) -> np.ndarray:
    """K bits (CRC included) -> E rate-matched coded bits (E >= N)."""
    c = np.asarray(c, dtype=np.uint8)
    k = len(c)
    u = np.zeros(n, dtype=np.uint8)
    u[info_set(k, n)] = c[input_interleaver(k)]
    d = polar_transform(u)
    y = d[subblock_index(n)]
    return y[np.arange(e) % n]


def encode_rm(c: np.ndarray, e: int, n_max: int = 9) -> np.ndarray:
    """K bits -> E bits with any E (repetition, puncturing or shortening),
    I_IL = 1, I_BIL = 0 (the DCI case)."""
    c = np.asarray(c, dtype=np.uint8)
    k = len(c)
    n = code_length(k, e, n_max)
    u = np.zeros(n, dtype=np.uint8)
    u[info_set_rm(k, n, e)] = c[input_interleaver(k)]
    y = polar_transform(u)[subblock_index(n)]
    if e >= n:
        return y[np.arange(e) % n]
    return y[n - e:] if _punctured(k, e) else y[:e]


def rate_dematch_rm(llr: np.ndarray, k: int, n_max: int = 9) -> np.ndarray:
    """Inverse of encode_rm: E soft bits -> N soft bits of d."""
    llr = np.asarray(llr, dtype=np.float64)
    e = len(llr)
    n = code_length(k, e, n_max)
    if e >= n:
        return rate_dematch(llr, n)
    y = np.zeros(n)
    if _punctured(k, e):
        y[n - e:] = llr                     # punctured bits: no information
    else:
        y[:e] = llr
        y[e:] = 1e4                         # shortened bits are known zeros
    d = np.empty(n)
    d[subblock_index(n)] = y
    return d


def rate_dematch(llr: np.ndarray, n: int) -> np.ndarray:
    """E soft bits -> N soft bits of d (repetitions are combined)."""
    llr = np.asarray(llr, dtype=np.float64)
    y = np.zeros(n)
    np.add.at(y, np.arange(len(llr)) % n, llr)
    d = np.empty(n)
    d[subblock_index(n)] = y
    return d


# ------------------------------------------------------------- SCL decoder
class _List:
    def __init__(self, frozen: np.ndarray, size: int):
        self.frozen = frozen                       # bool per leaf
        self.size = size
        n = len(frozen)
        self.u = np.zeros((size, n), dtype=np.uint8)
        self.pm = np.full(size, 1e12)
        self.pm[0] = 0.0
        # frozen-only (rate-0) subtrees, keyed by (start, length)
        self.rate0 = {}

    def is_rate0(self, start, length):
        key = (start, length)
        r = self.rate0.get(key)
        if r is None:
            r = bool(self.frozen[start:start + length].all())
            self.rate0[key] = r
        return r


def _f(a, b):
    return np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))


def _node(st: _List, alpha: np.ndarray, start: int):
    """alpha: (L, n) LLRs of this node. Returns (beta (L, n), perm (L,))."""
    L, n = alpha.shape
    ident = np.arange(L)
    if st.is_rate0(start, n):
        st.pm += np.sum(np.where(alpha < 0, -alpha, 0.0), axis=1)
        return np.zeros((L, n), dtype=np.uint8), ident
    if n == 1:
        a = alpha[:, 0]
        # information bit: fork every path into u=0 / u=1
        pm0 = st.pm + np.where(a < 0, -a, 0.0)
        pm1 = st.pm + np.where(a > 0, a, 0.0)
        cand = np.concatenate([pm0, pm1])
        keep = np.argsort(cand, kind="stable")[:L]
        parent = keep % L
        bit = (keep >= L).astype(np.uint8)
        st.u = st.u[parent]
        st.u[:, start] = bit
        st.pm = cand[keep]
        return bit[:, None], parent
    h = n // 2
    a, b = alpha[:, :h], alpha[:, h:]
    beta_l, p1 = _node(st, _f(a, b), start)
    a, b = a[p1], b[p1]
    beta_r, p2 = _node(st, b + (1 - 2 * beta_l.astype(np.float64)) * a, start + h)
    beta_l = beta_l[p2]
    return np.concatenate([beta_l ^ beta_r, beta_r], axis=1), p1[p2]


def decode(llr_d: np.ndarray, k: int, check, list_size: int = 8, e: int | None = None):
    """CRC-aided SCL decoding. llr_d: N soft bits of d (after
    rate_dematch); check(c) -> bool tests the K decoded bits (CRC); e: the
    rate-matched length when it may be below N (frozen set).
    Returns the K bits c of the best path that passes `check`, else None."""
    n = len(llr_d)
    info = info_set(k, n) if e is None else info_set_rm(k, n, e)
    frozen = np.ones(n, dtype=bool)
    frozen[info] = False
    st = _List(frozen, list_size)
    alpha = np.tile(np.asarray(llr_d, dtype=np.float64), (list_size, 1))
    _node(st, alpha, 0)
    pi = input_interleaver(k)
    for i in np.argsort(st.pm, kind="stable"):
        if st.pm[i] >= 1e11:
            break
        c = np.empty(k, dtype=np.uint8)
        c[pi] = st.u[i, info]
        if check(c):
            return c
    return None
