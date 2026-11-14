# ltescan — skaner i analizator LTE (warstwa PHY) dla HackRF

*English version: [README.md](README.md)*

Pasywny odbiornik łącza w dół (downlink) LTE napisany w czystym Pythonie (NumPy/SciPy).
Nie jest to pełny stos 3GPP. Narzędzie demoduluje warstwę fizyczną i rozumie kanały
rozgłoszeniowe oraz sterujące:

| Warstwa | Co jest zaimplementowane |
|---|---|
| Synchronizacja | PSS (3 korzenie Zadoffa-Chu, przeszukiwanie CFO), SSS (168 grup), wykrywanie FDD/TDD i CP normalnego/rozszerzonego, granica ramki, CFO zgrubne/dokładne (połówki PSS + CP), śledzenie dryfu zegara próbkowania |
| Wiele komórek | iteracyjne kasowanie PSS/SSS (SIC), odrzucanie fantomów (margines SSS, wspólny CFO, rozstrzyganie niejednoznaczności ZC ±k·15 kHz) |
| OFDM | 1.4/3/5/10/15/20 MHz (6–100 RB), CRS dla 1/2/4 portów, estymacja kanału LS + interpolacja |
| Pomiary | RSRP (dBFS/RE), RSRQ, SINR na CRS, CFO i błąd zegara odbiornika w ppm |
| PBCH → **MIB** | szerokość pasma, liczba portów (maska CRC), konfiguracja PHICH, pełny SFN; dekodowanie z jednej ramki lub łączenie miękkie 4 ramek |
| PCFICH | CFI (1–3) |
| PHICH | rozmieszczenie REG (potrzebne do PDCCH; łącznie z m_i dla TDD) |
| PDCCH | przeplot czwórek, CCE, ślepe dekodowanie przestrzeni wspólnej (DCI 0/1A, 1C) dla SI-/P-/RA-RNTI; opcjonalnie wykrywanie aktywnych C-RNTI (DCI 0/1A, 1) i zajętość CCE |
| PDSCH | alokacja typu 2 (zlokalizowana i rozproszona VRB→PRB), TBS, QPSK, dywersyfikacja nadawcza SFBC, dopasowanie szybkości, **dekoder turbo** (max-log-MAP), CRC24A, łączenie HARQ wersji RV |
| RRC (ASN.1 UPER) | **SIB1**: PLMN (MCC-MNC), TAC, Cell Identity (eNB ID + sektor), pasmo, cellBarred, q-RxLevMin, p-Max, harmonogram SI, konfiguracja TDD; wiadomości SI: lista zawartych SIB |
| Skaner | szybkie przechwyty 20 MHz, wyszukiwanie nośnych po widmie, test kandydatów na rastrze 100 kHz, uczenie się błędu ppm HackRF |

Do tego wbudowany jest **generator sygnału eNB** (`ltescan synth`) z modelem kanału
(szum, CFO, dryf zegara, wielodrogowość). Na nim opierają się testy, a bez sprzętu
można przejść cały łańcuch od początku do końca.

## Instalacja

```bash
python -m venv .venv
.venv\Scripts\activate          # Linux/macOS: source .venv/bin/activate
pip install -e .[plot,test]
```

### HackRF

`ltescan` steruje HackRF na jeden z dwóch sposobów:

1. **`hackrf_transfer`** (domyślnie): program z pakietu *HackRF host tools* musi być w `PATH`.
   Na Windows najprościej zainstalować go z **radioconda** albo z instalatora **PothosSDR**.
   Urządzenie potrzebuje sterownika WinUSB (instalują go te pakiety albo Zadig).
   Sprawdzenie: `hackrf_info`.
2. **SoapySDR** (`--driver soapy`): wymaga modułów Pythona `SoapySDR` i `SoapyHackRF`.

Bez HackRF wszystkie polecenia działają na plikach IQ (`--file`).

## Szybki start

```bash
ltescan bands                         # obsługiwane pasma i zakresy EARFCN
ltescan earfcn 1575 6300 2655M        # EARFCN <-> częstotliwość

# skan pasm (np. polskie: 1, 3, 7, 8, 20, 28, 38)
ltescan scan --band 3 --band 20 --json skan.json --plot widmo.png

# skan znanych nośnych (+ SIB1: PLMN, TAC, Cell ID)
ltescan scan --earfcn 1575 --earfcn 6300 --sib

# szczegółowa analiza jednej nośnej
ltescan cell --earfcn 1575 --rnti --dci 20 --plot wykresy --json komorka.json

# nagranie IQ i późniejsza analiza offline
ltescan record --earfcn 1575 --rate 15.36e6 --duration 0.2 --out b3.cs8
ltescan cell --file b3.cs8
```

Bez sprzętu:

```bash
ltescan synth --out test.cs8 --nrb 50 --pci 260 --ports 2 --snr 10 --cfo 23e3 --rate 15.36e6
ltescan cell --file test.cs8 --rnti --dci 6 --plot wykresy
```

Przykładowy wynik (`cell`; komunikaty programu są po angielsku):

```
Cell PCI 260 (N_ID1=86, N_ID2=2) @ 1842.500 MHz, EARFCN 1575, band 3
Duplex: FDD, CP: normal
Sync: PSS metric 0.98 (PMR 133), SSS metric 1.00 (margin 2.07)
CFO: +22976 Hz, receiver clock error ~ -12.47 ppm
RSRP: -36.9 dBFS/RE, RSRQ: -8.0 dB, SINR(CRS): 25.9 dB
MIB: bandwidth 10 MHz (50 RB), CRS ports: 2, PHICH: normal, Ng=1/2, SFN=0
Analyzed 118 subframes; CFI2: 118; PDCCH CCE load ~ 16%
SIB1:
  PLMN: 260-03
  TAC: 6699 (0x1a2b)
  Cell Identity: 3160327 (eNB 12345, sector 7)
  Band (freqBandIndicator): 3
  ...
  SI-1: every 16 frames: SIB2
  SI-2: every 32 frames: SIB3, SIB5
Active C-RNTIs (2): 0x1234x118, 0x4321x118
```

### Praktyczne wskazówki dla HackRF

- **Wzmocnienie**: zacznij od `--lna 24 --vga 20`. `record` ostrzega przed przesterowaniem.
  Za mocny sygnał (np. blisko stacji) zatyka 8-bitowy przetwornik, za słaby nie przechodzi progu PSS.
- **Zegar**: kwarc HackRF One ma tolerancję ±20 ppm, czyli do ±52 kHz na 2.6 GHz.
  Domyślny zakres szukania CFO wynosi 20 ppm częstotliwości. Raport podaje oszacowany
  błąd zegara. Podaj go potem jako `--ppm` (korekcja strojenia) albo `--ppm-hint`
  (zawężenie szukania), a skan przyspieszy i zyska na czułości.
- **Próbkowanie**: `cell` domyślnie używa 15.36 Msps, co wystarcza dla komórek ≤10 MHz.
  Gdy MIB pokaże 15/20 MHz, przechwyt jest automatycznie powtarzany z 20 Msps
  (maksimum HackRF). Komórka 20 MHz zajmuje 18 MHz, więc skrajne podnośne leżą na
  zboczu filtra. PDCCH zwykle się dekoduje, bo jest rozproszony po całym paśmie i
  chroniony kodem splotowym.
- **Antena**: dookólna antena na pasma LTE. Typowa teleskopowa z zestawu HackRF działa
  przy dobrym zasięgu.

## Jak to działa

```
IQ (HackRF / plik)
 └─ przepróbkowanie do 1.92 Msps (środkowe 6 RB)
     ├─ PSS: korelacja z 3 wzorcami × siatka CFO, akumulacja co 5 ms
     │     → hipotezy CFO ±k·15 kHz (niejednoznaczność ZC) rozstrzygane przez SSS
     ├─ CFO dokładne: faza między połówkami PSS, następnie korelacja CP
     ├─ SSS: korekcja kanałem z PSS, 336 kandydatów × {FDD,TDD}×{CP norm,ext}
     │     → PCI, granica ramki; SIC i ponowne szukanie słabszych komórek
     └─ PBCH: CRS → estymata kanału, SFBC, deskrambling (Gold, c_init=PCI),
           odwrotne dopasowanie szybkości, Viterbi z zawijaniem (tail-biting),
           CRC16 z maską portów → MIB
 └─ przepróbkowanie do częstotliwości komórki (np. 15.36 Msps dla 50 RB)
     ├─ śledzenie PSS co 5 ms → regresja liniowa położenia (dryf zegara próbkowania)
     └─ dla każdej podramki: FFT → CRS/kanał → PCFICH (CFI) → PDCCH:
           REG-i (bez PCFICH/PHICH) → odwrócenie przesunięcia i przeplotu czwórek
           → CCE → ślepe dekodowanie (Viterbi + CRC16 XOR RNTI)
           → DCI 1A/1C dla SI-RNTI → PDSCH: VRB→PRB, mapowanie RE (bez CRS,
           PBCH, PSS/SSS i regionu sterującego), SFBC, deskrambling, turbo, CRC24A
           → ASN.1 UPER → SIB1 / SI
```

Wykrywanie C-RNTI dekoduje każdą pozycję CCE na wszystkich poziomach agregacji.
RNTI wynika z reszty CRC, więc pasuje „każda” pozycja. Prawdziwe kandydaty oddziela
od szumu metryka: ponowne zakodowanie zdekodowanych bitów i ważona |LLR| zgodność z
odebranymi bitami miękkimi. Losowe dane osiągają maks. ~0.95 / 0.83 / 0.73 / 0.67 dla
agregacji 1/2/4/8. Raport pokazuje tylko RNTI widziane co najmniej 2 razy.

Słabsze komórki znalezione po kasowaniu interferencji, dla których nie udało się
zdekodować MIB, są wypisywane w jednej linii jako niepotwierdzone (mogą być resztkami
najsilniejszej komórki). `--all` pokazuje dla nich pełne raporty.

## Struktura

```
ltescan/
  params.py     numerologia, pasma, EARFCN, konfiguracje TDD
  sequences.py  ciąg Golda, PSS, SSS, CRS
  ofdm.py       (de)modulacja OFDM, CP
  dsp.py        przepróbkowanie, mieszanie, PSD
  sync.py       szukanie komórek: PSS/SSS/CFO/SIC
  chest.py      estymacja kanału, RSRP/RSRQ/SINR
  mimo.py       SFBC (2 i 4 porty)
  coding.py     CRC, kod splotowy + Viterbi, przeplatacz sub-blokowy, rate matching
  turbo.py      kod turbo (QPP), dekoder max-log-MAP, rate matching, bufor HARQ
  pbch.py       PBCH/MIB
  control.py    REG, PCFICH, PHICH, PDCCH, przestrzenie wyszukiwania
  dci.py        formaty DCI 0/1A/1C/1, RIV
  pdsch.py      TBS, VRB→PRB, mapowanie RE, dekodowanie PDSCH
  rrc.py        ASN.1 UPER: SIB1, SystemInformation
  receiver.py   odbiornik: MIB → PDCCH → SIB, statystyki
  scanner.py    skaner pasm
  sdr.py        HackRF (hackrf_transfer / SoapySDR), pliki IQ (.cs8/.cs16/.cf32/.npy)
  synth.py      generator sygnału eNB + model kanału
  report.py     raporty tekstowe/JSON, wykresy
  cli.py        interfejs wiersza poleceń
tests/          49 testów (jednostkowe + end-to-end na sygnale syntetycznym)
```

Testy: `pytest -q` (~35 s).

## Weryfikacja i ograniczenia

**Zweryfikowane niezależnie:**

- wartość kontrolna CRC16;
- tabele QPP turbo: źródła srsRAN plus sprawdzenie, że każda z 188 daje permutację;
- kolumny TBS dla N_PRB = 2/3 (srsRAN);
- rozmiary DCI dla wszystkich szerokości pasma;
- konwersje EARFCN.

**Przetestowane na sygnale syntetycznym:**

- wszystkie szerokości pasma, 1/2/4 porty, FDD i TDD, oba CP;
- DCI 1A i 1C, alokacja zlokalizowana i rozproszona;
- SNR do −8 dB, CFO ±40 kHz, dryf ±20 ppm, wielodrogowość;
- nienatywne częstotliwości próbkowania (10–20 Msps);
- dwie komórki na jednej nośnej (sąsiad 26 dB słabszy);
- losowy przegląd 50 konfiguracji, wszystkie poprawne, bez fantomów;
- skan pasma przez atrapę `hackrf_transfer`.

**Nie przetestowano na prawdziwej sieci.** Generator i odbiornik korzystają z tej samej
interpretacji specyfikacji. Błąd w rzadziej używanych szczegółach mógłby się więc w
testach znosić. Dotyczy to:

- rozmieszczenia REG PHICH (zwłaszcza rozszerzony czas trwania);
- mapowania rozproszonych VRB;
- wyłączeń RE PDSCH wokół PSS/SSS/PBCH przy nieparzystej liczbie RB (15/25/75);
- tabeli TBS dla DCI 1C (z pamięci, bez drugiego źródła).

Synchronizacja, MIB i PCFICH są odporne na takie problemy. Gdyby na prawdziwym sygnale
MIB działał, a SIB1 nie, warto zacząć diagnozę od tych miejsc.

Pozostałe ograniczenia:

- tylko transport bloki z jednym blokiem kodowym (≤ 6120 bitów), co wystarcza dla SI;
- brak dekodowania PDSCH użytkowników, PMCH/MBSFN, PHICH (ACK/NACK);
- SIB2 i dalsze: rozpoznawany jest tylko typ, bez treści;
- TDD: konfiguracja UL/DL jest znana dopiero z SIB1. Wcześniej analizowane są tylko
  podramki 0 i 5, a podramki specjalne są pomijane;
- wydajność czystego Pythona: około 10–20 ms na podramkę (ślepe dekodowanie
  C-RNTI wolniej), to analiza offline, nie czas rzeczywisty;
- RSRP podawane w dBFS. Wartość w dBm wymaga kalibracji toru odbiorczego.

## Uwagi prawne

Narzędzie odbiera pasywnie i dekoduje wyłącznie informacje **rozgłoszeniowe** (MIB/SIB),
nadawane jawnie do wszystkich terminali. Nie dekoduje treści transmisji użytkowników
ani wiadomości paging (pokazuje tylko, że paging wystąpił). Wykrywane C-RNTI to
tymczasowe identyfikatory radiowe, niepowiązane z tożsamością abonenta. Odbiór i
analiza sygnałów radiowych podlegają lokalnym przepisom (w Polsce m.in. tajemnica
komunikacji w Prawie komunikacji elektronicznej). Narzędzie jest przeznaczone do
celów edukacyjnych, badawczych i pomiarowych.

## Autorzy

- Przemysław Frasunek <przemyslaw@frasunek.com>
- Claude (Anthropic), przez Claude Code

## Licencja

MIT, zob. [LICENSE](LICENSE).
