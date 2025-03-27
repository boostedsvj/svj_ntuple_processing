import os, os.path as osp, logging, pprint, uuid, re
from contextlib import contextmanager
from collections import OrderedDict
import copy
from typing import Callable, List, Tuple
import inspect

import numpy as np
import uproot
import awkward as ak

INCLUDE_DIR = osp.join(osp.abspath(osp.dirname(__file__)), "include")
def version():
    with open(osp.join(INCLUDE_DIR, "VERSION"), "r") as f:
        return(f.read().strip())

def uid():
    return str(uuid.uuid4())


def setup_logger(name='svj'):
    if name in logging.Logger.manager.loggerDict:
        logger = logging.getLogger(name)
        logger.info('Logger %s is already defined', name)
    else:
        fmt = logging.Formatter(
            fmt = (
                '\033[92m[%(name)s:%(levelname)s:%(asctime)s:%(module)s:%(lineno)s]\033[0m'
                + ' %(message)s'
                ),
            datefmt='%Y-%m-%d %H:%M:%S'
            )
        handler = logging.StreamHandler()
        handler.setFormatter(fmt)
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        logger.addHandler(handler)
    return logger
logger = setup_logger()

triggers_per_year = {}

triggers_per_year[2018] = {
    'jetht': [
        # AK8PFJet triggers
        'HLT_AK8PFJet500_v',
        'HLT_AK8PFJet550_v',
        # CaloJet
        'HLT_CaloJet500_NoJetID_v',
        'HLT_CaloJet550_NoJetID_v',
        # PFJet and PFHT
        'HLT_PFHT1050_v',
        'HLT_PFJet500_v',
        'HLT_PFJet550_v',
        # Trim mass jetpt+HT
        'HLT_AK8PFHT800_TrimMass50_v',
        'HLT_AK8PFHT850_TrimMass50_v',
        'HLT_AK8PFHT900_TrimMass50_v',
        'HLT_AK8PFJet400_TrimMass30_v',
        'HLT_AK8PFJet420_TrimMass30_v',
    ],
    'htmht': [
        # MET triggers
        'HLT_PFHT500_PFMET100_PFMHT100_IDTight_v',
        'HLT_PFHT500_PFMET110_PFMHT110_IDTight_v',
        'HLT_PFHT700_PFMET85_PFMHT85_IDTight_v',
        'HLT_PFHT700_PFMET95_PFMHT95_IDTight_v',
        'HLT_PFHT800_PFMET75_PFMHT75_IDTight_v',
        'HLT_PFHT800_PFMET85_PFMHT85_IDTight_v',
    ],
}

triggers_per_year[2017] = copy.deepcopy(triggers_per_year[2018])
triggers_per_year[2017]['jetht'].append('HLT_AK8PFJet360_TrimMass30_v')

triggers_per_year[2016] = {
    'jetht': [
        'HLT_AK8PFHT700_TrimR0p1PT0p03Mass50_v',
        'HLT_AK8PFJet360_TrimMass30_v',
        'HLT_CaloJet500_NoJetID_v',
        'HLT_PFHT900_v',
        'HLT_PFJet450_v',
        'HLT_PFJet500_v',
        'HLT_PFHT800_v',
    ],
    'htmht': [
        # MET
        'HLT_PFHT300_PFMET100_v',
        'HLT_PFHT300_PFMET110_v',
    ],
}

trigger_hierarchy = [
    'jetht',
    'htmht',
]

# to prevent overlap in PDs:
# take events in PD0 passing any trigger
# take events in PD1 passing any trigger but failing PD0-specific triggers
# take events in PD2 passing any trigger but failing PD0,PD1-specific triggers
# etc.
# (for MC, just take all triggers)
def get_pd_triggers(pd,year):
    good_triggers = []
    bad_triggers = []
    pd_index = 0
    if pd in trigger_hierarchy:
        pd_index = trigger_hierarchy.index(pd)

    for key,val in triggers_per_year[year].items():
        key_index = 0
        if pd in trigger_hierarchy:
            key_index = trigger_hierarchy.index(key)

        if key_index < pd_index:
            bad_triggers += val
        else:
            good_triggers += val
    return good_triggers, bad_triggers


#  ECAL DEAD CELL LOCATIONS
# ecaldeadcells 2018 remove 5sigma
dataqcd_eta_ecaldead={}
dataqcd_phi_ecaldead={}
dataqcd_eta_ecaldead[2018] = np.array([
    -1.632, -0.768, -2.112, -1.632, -0.384,  0.864, -1.536,  1.056,
    1.44 , -2.016,  1.824, -2.4  , -1.44 , -0.192, -0.864, -1.536,
    0.   , -1.248,  1.152, -0.288, -2.304,  1.632,  1.728, -2.4  ,
    -0.768,  1.248,  0.96 , -1.728, -0.192, -2.304, -0.096,  1.536,
    -2.208,  0.096,  0.864, -0.96 , -0.576,  1.728, -1.248
    ])
dataqcd_phi_ecaldead[2018] = np.array([
    0.503, -0.754, -2.513,  0.628,  0.126,  2.765, -3.142, -3.142,
    -0.251, -2.513, -2.136, -1.508, -3.142, -2.639, -1.759, -1.257,
    1.759, -1.257, -0.503,  0.126, -1.508, -0.628, -0.628, -1.634,
    -2.262, -0.503,  0.628, -0.503,  0.628, -1.634,  0.88 , -1.634,
    -1.508,  1.759,  1.634,  2.011,  2.639, -2.136,  2.765
    ])


dataqcd_eta_ecaldead[2017] = np.array([
        1.344, -1.632, -1.632, -2.112, -0.384,  1.152,  0.864, -1.536,
        1.344,  1.056, -1.152, -2.016, -0.48 ,  1.824, -1.344, -1.44 ,
       -2.4  , -0.192, -0.864,  0.   , -1.248, -0.288, -2.304,  1.632,
       -0.768,  1.248,  0.96 , -1.728, -0.192, -0.096,  0.768,  1.728,
        1.536,  1.44 , -2.208, -2.208,  0.096, -0.96 ,  0.864, -0.576,
        1.728])
dataqcd_phi_ecaldead[2017] = np.array([
        2.639,  0.503,  0.628, -2.513,  0.126,  0.   ,  2.765, -3.142,
        2.765, -3.142,  0.503, -2.513,  2.639, -2.136, -2.765, -3.142,
       -1.508, -2.639, -1.759,  1.759, -1.257,  0.126, -1.508, -0.628,
       -2.262, -0.503,  0.628, -0.503,  0.628,  0.88 ,  2.765,  0.754,
       -1.634,  2.639, -2.513, -1.508,  1.759,  2.011,  1.634,  2.639,
       -2.136])

dataqcd_eta_ecaldead[2016] = np.array([
       -0.768, -1.536,  1.44 , -1.152, -2.016, -0.48 ,  1.824, -2.4  ,
       -1.248, -2.304,  1.632,  1.344, -0.768,  1.248,  2.304, -0.192,
        1.632,  1.536,  1.44 ,  1.344,  0.096,  0.864, -0.96 , -0.576,
       -1.632, -1.632, -2.112, -0.384,  0.864, -1.632,  1.056, -1.344,
       -1.44 , -0.192, -0.864, -0.   , -0.288, -0.96 ,  1.728,  0.96 ,
       -1.728, -0.096,  0.192,  1.728, -1.152, -2.208,  1.728])
dataqcd_phi_ecaldead[2016] = np.array([
       -0.754, -3.142, -0.251,  0.503, -2.513,  2.639, -2.136, -1.508,
       -1.257, -1.508, -0.628, -0.251, -2.262, -0.503, -2.765,  0.628,
        0.754, -1.634,  2.639, -2.639,  1.759,  1.634,  2.011,  2.639,
        0.503,  0.628, -2.513,  0.126,  2.765,  2.513, -3.142, -2.765,
       -3.142, -2.639, -1.759,  1.759,  0.126,  2.89 , -0.628,  0.628,
       -0.503,  0.88 , -2.513,  0.754, -1.257, -1.508, -2.136])


def metadata_from_path(path):
    """
    Extracts metadata from a filename.
    Not very robust but not much else to go on either.
    """
    meta = {}
    fullpath = path
    path = osp.basename(path)
    if path[0].isdigit():
        # follows TreeMaker naming convention (folderized)
        sample_path = osp.basename(osp.dirname(fullpath))
        year_path = osp.basename(osp.dirname(osp.dirname(fullpath)))
    else:
        # older sample
        year_path = fullpath
        sample_path = path

    match = re.search(r'year(\d+)', year_path)
    if match:
        meta['year'] = int(match.group(1))

    match = re.search(r'UL(?:20)?(\d\d)', year_path)
    if match:
        meta['year'] = int('20'+match.group(1))

    if 'year' not in meta:
        meta['year'] = 2018

    pds = ['jetht','htmht','met','singlemuon']
    for pd in pds:
        if sample_path.lower().startswith(pd):
            meta['sample_type'] = 'data'
            meta['data_type'] = pd
            break

    bkgs = ['qcd', 'ttjets', 'wjets', 'zjets']
    for bkg in bkgs:
        if sample_path.lower().startswith(bkg):
            meta['sample_type'] = 'bkg'
            meta['bkg_type'] = bkg
            break

    if 'sample_type' not in meta:
        meta['sample_type'] = 'sig'

    # bkg-specific info
    if meta['sample_type']=='bkg':
        if 'HT' in sample_path:
            match = re.search(r'HT\-(\d+)[tT]o([\dInf]+)', sample_path)
            meta['htbin'] = [float(match.group(1)), float(match.group(2))]
        elif 'Pt' in sample_path:
            match = re.search(r'Pt_(\d+)to([\dInf]+)', sample_path)
            meta['ptbin'] = [float(match.group(1)), float(match.group(2))]

        if meta['bkg_type'] == 'ttjets':
            if 'SingleLep' in sample_path:
                meta['n_lepton_sample'] = 1
            elif 'DiLep' in sample_path:
                meta['n_lepton_sample'] = 2

        if 'genMET' in sample_path:
            meta['genmet_sample'] = True

    # sig-specific info
    elif meta['sample_type']=='sig':
        match = re.search(r'madpt(\d+)', sample_path)
        if match: meta['madpt'] = int(match.group(1))

        match = re.search(r'MADPT(\d+)', sample_path)
        if match: meta['madpt'] = int(match.group(1))

        match = re.search(r'genjetpt(\d+)', sample_path)
        if match:
            meta['genjetpt'] = int(match.group(1))

        match = re.search(r'mz(\d+)', sample_path)
        if match: meta['mz'] = int(match.group(1))

        match = re.search(r'mMed-(\d+)', sample_path)
        if match: meta['mz'] = int(match.group(1))

        match = re.search(r'mdark(\d+)', sample_path)
        if match: meta['mdark'] = int(match.group(1))

        match = re.search(r'mDark-(\d+)', sample_path)
        if match: meta['mdark'] = int(match.group(1))

        match = re.search(r'rinv(\d(\.\d+)?)', sample_path)
        if match: meta['rinv'] = float(match.group(1))

        match = re.search(r'rinv-(\d(\.\d+)?)', sample_path)
        if match: meta['rinv'] = float(match.group(1))

        match = re.search(r'rinv-(\d(p\d+)?)', sample_path)
        if match: meta['rinv'] = float(match.group(1).replace('p','.'))

    return meta


class Arrays:
    """
    Container class for an awkward.Array object + metadata about it.

    The functions filter_preselection, filter_zprime_in_cone, and filter_stitch
    can be used on this object to apply event selections.

    This object is meant to be converted to a Columns object before being
    saved to disk.
    """
    def __init__(self, array=None):
        self.array = array
        self.trigger_branch = None
        self.cutflow = OrderedDict()
        self.metadata = {}
        self._xs = None

    def __len__(self):
        return len(self.array)

    def __repr__(self):
        return '<Arrays {0}>'.format(pprint.pformat(self.metadata))

    def cut(self, cut_name):
        """Adds an entry to the cutflow list now"""
        self.cutflow[cut_name] = len(self)

    def copy(self):
        copy_ = Arrays(copy.copy(self.array))
        copy_.trigger_branch = self.trigger_branch
        copy_.cutflow = self.cutflow.copy()
        copy_.metadata = self.metadata.copy()
        return copy_

    @property
    def year(self):
        return self.metadata['year']

    @property
    def bkg_type(self):
        if 'bkg_type' in self.metadata:
            return self.metadata['bkg_type']

    @property
    def data_type(self):
        if 'data_type' in self.metadata:
            return self.metadata['data_type']

    @property
    def triggers(self):
        return get_pd_triggers(self.data_type,self.year)

    @property
    def xs(self):
        if 'bkg_type' in self.metadata:
            raise NotImplementedError('XS not gettable for background')
        if self._xs is None:
            # Load the signal cross section polynomial
            import requests
            fit = np.poly1d(
                requests
                .get('https://raw.githubusercontent.com/boostedsvj/svj_madpt_crosssection/main/fit_madpt300.txt')
                .json()
                )
            self._xs = fit(self.metadata['mz'])
        return self._xs


@contextmanager
def local_copy(remote,bypass=False):
    """
    Creates a temporary local copy of a remote file
    """
    import seutils
    must_delete = False
    try:
        if bypass:
            yield remote
        elif seutils.path.has_protocol(remote):
            # File is remote, make local copy
            must_delete = True
            local = uid() + osp.splitext(remote)[1]
            logger.debug('Copying %s -> %s', remote, local)
            seutils.cp(remote, local)
            yield local
        else:
            # File is already local, nothing to do
            yield remote
    finally:
        if must_delete:
            try:
                os.remove(local)
            except Exception:
                pass


BRANCHES = [
    'RunNum','LumiBlockNum','EvtNum',
    'Jets.fCoordinates.fPt', 'Jets.fCoordinates.fEta',
    'Jets.fCoordinates.fPhi', 'Jets.fCoordinates.fE',
    'JetsAK8.fCoordinates.fPt', 'JetsAK8.fCoordinates.fEta',
    'JetsAK8.fCoordinates.fPhi', 'JetsAK8.fCoordinates.fE',
    'JetsAK15.fCoordinates.fPt', 'JetsAK15.fCoordinates.fEta',
    'JetsAK15.fCoordinates.fPhi', 'JetsAK15.fCoordinates.fE',
    'JetsAK15_ecfC2b1', 'JetsAK15_ecfC2b2',
    'JetsAK15_ecfD2b1', 'JetsAK15_ecfD2b2',
    'JetsAK15_ecfM2b1', 'JetsAK15_ecfM2b2',
    'JetsAK15_ecfN2b1', 'JetsAK15_ecfN2b2',
    'JetsAK15_girth', 'JetsAK15_ptD',
    'JetsAK15_axismajor', 'JetsAK15_axisminor',
    'JetsAK15_chargedHadronEnergyFraction', 'JetsAK15_electronEnergyFraction', 'JetsAK15_muonEnergyFraction',
    'JetsAK15_neutralHadronEnergyFraction', 'JetsAK15_photonEnergyFraction',
    'HT',
    'MET', 'METPhi',
    'TriggerPass',
    'JetsAK15_ID',
    'NMuons', 'NElectrons',
    'HBHENoiseFilter', 'HBHEIsoNoiseFilter', 'eeBadScFilter', 'PrimaryVertexFilter',
    'ecalBadCalibFilter', 'hfNoisyHitsFilter',
    'BadPFMuonFilter', 'BadPFMuonDzFilter', 'globalSuperTightHalo2016Filter',
    # highMET events
    'CaloMET', 'PFCaloMETRatio',
    # Muon stuff
    'Muons.fCoordinates.fPt', 'Muons.fCoordinates.fEta',
    'Muons.fCoordinates.fPhi', 'Muons.fCoordinates.fE',
    'Muons_iso', 'Muons_mediumID',
    # more new branches from Kevin 05/06/2024
    'JetsAK15_nConstituents','JetsAK15_nConstituentsSoftDrop',
    # new ecf branches
    'JetsAK15_ecfFullC2b1','JetsAK15_ecfFullC2b2','JetsAK15_ecfFullD2b1','JetsAK15_ecfFullD2b2','JetsAK15_ecfFullM2b1',
    'JetsAK15_ecfFullM2b2','JetsAK15_ecfFullN2b1','JetsAK15_ecfFullN2b2'
    ]

BRANCHES_HLT = [
    # HLT
    'HLTMuonObjects.fCoordinates.fPt', 'HLTMuonObjects.fCoordinates.fEta',
    'HLTMuonObjects.fCoordinates.fPhi', 'HLTMuonObjects.fCoordinates.fE',
    ]

BRANCHES_GENONLY = [
    'Weight', 'puWeight',
    'madHT', 'GenMET',
    'GenParticles_PdgId',
    'GenParticles_Status',
    'GenParticles.fCoordinates.fPt',
    'GenParticles.fCoordinates.fEta',
    'GenParticles.fCoordinates.fPhi',
    'GenParticles.fCoordinates.fE',
    'GenElectrons.fCoordinates.fPt',
    'GenMuons.fCoordinates.fPt',
    'GenTaus.fCoordinates.fPt',
    'ScaleWeights',
    'PDFweights', 'PSweights',
    'puSysUp', 'puSysDown',
    'METDown', 'METUp', 'METPhiDown', 'METPhiUp',
    'JetsAK15_jecFactor','JetsAK15_jecUnc','JetsAK15_jerFactor',
    'JetsAK15_jerFactorDown','JetsAK15_jerFactorUp','JetsAK15JECdown_jerFactor','JetsAK15JECdown_origIndex',
    'JetsAK15JECup_jerFactor','JetsAK15JECup_origIndex','JetsAK15JERdown_origIndex','JetsAK15JERup_origIndex',
    ]

BRANCHES_JERJEC = [
    'JetsJECdown_jerFactor',
    'JetsJECdown_origIndex',
    'JetsJECup_jerFactor',
    'JetsJECup_origIndex',
    'JetsJERdown_origIndex',
    'JetsJERup_origIndex',
    'Jets_origIndex',
    'Jets_jecFactor',
    'Jets_jecUnc',
    'Jets_jerFactor',
    'Jets_jerFactorDown',
    'Jets_jerFactorUp',
    'JetsAK8JECdown_jerFactor',
    'JetsAK8JECdown_origIndex',
    'JetsAK8JECup_jerFactor',
    'JetsAK8JECup_origIndex',
    'JetsAK8JERdown_origIndex',
    'JetsAK8JERup_origIndex',
    'JetsAK8_origIndex',
    'JetsAK8_jecFactor',
    'JetsAK8_jecUnc',
    'JetsAK8_jerFactor',
    'JetsAK8_jerFactorDown',
    'JetsAK8_jerFactorUp',
    'JetsAK15JECdown_jerFactor',
    'JetsAK15JECdown_origIndex',
    'JetsAK15JECup_jerFactor',
    'JetsAK15JECup_origIndex',
    'JetsAK15JERdown_origIndex',
    'JetsAK15JERup_origIndex',
    'JetsAK15_origIndex',
    'JetsAK15_jecFactor',
    'JetsAK15_jecUnc',
    'JetsAK15_jerFactor',
    'JetsAK15_jerFactorDown',
    'JetsAK15_jerFactorUp',
    'GenJets.fCoordinates.fPt',
    'GenJets.fCoordinates.fEta',
    'GenJets.fCoordinates.fPhi',
    'GenJets.fCoordinates.fE',
    'GenJetsAK8.fCoordinates.fPt',
    'GenJetsAK8.fCoordinates.fEta',
    'GenJetsAK8.fCoordinates.fPhi',
    'GenJetsAK8.fCoordinates.fE',
    'GenJetsAK15.fCoordinates.fPt',
    'GenJetsAK15.fCoordinates.fEta',
    'GenJetsAK15.fCoordinates.fPhi',
    'GenJetsAK15.fCoordinates.fE',
    ]

def open_root(rootfile, local=False, load_hlt=False):
    """
    Returns an Arrays object from a rootfile (unfiltered).
    """
    # get metadata first to determine branches to load
    metadata = metadata_from_path(rootfile)
    load_gen = metadata["sample_type"]!="data"
    load_jerjec = metadata["sample_type"]=="sig"

    branches = BRANCHES[:]
    # Only available for simulation, not data
    if load_gen: branches.extend(BRANCHES_GENONLY)
    if load_hlt: branches.extend(BRANCHES_HLT)
    if load_jerjec: branches.extend(BRANCHES_JERJEC)

    with local_copy(rootfile,bypass=not local) as fname:
        tree = uproot.open(fname + ':TreeMaker2/PreSelection')
        arrays = Arrays(tree.arrays(branches))

    # Store the order of trigger names in the array object
    arrays.trigger_branch = tree['TriggerPass'].title.split(',')
    arrays.metadata['src'] = rootfile
    arrays.metadata.update(metadata)
    arrays.cut('raw')
    return arrays


def calc_dphi(phi1, phi2):
    """
    Calculates delta phi. Assures output is within -pi .. pi.
    """
    twopi = 2.*np.pi
    # Map to 0..2pi range
    dphi = (phi1 - phi2) % twopi
    # Map pi..2pi --> -pi..0
    dphi[dphi > np.pi] -= twopi
    return dphi


def calculate_mass(pt, eta, e):
    """
    compute mass
    """
    pz = pt * np.sinh(eta)
    mass = np.sqrt(e**2 - pt**2 - pz**2)
    return mass

def calculate_rho(pt, eta, e):
    mass = calculate_mass(pt, eta, e)
    rho  = np.log(mass**2 / pt**2)
    return rho


def calc_dr(eta1, phi1, eta2, phi2):
    return np.sqrt((eta1-eta2)**2 + calc_dphi(phi1, phi2)**2)


def calc_rt(met, pt):
    return np.sqrt(1.+ met/pt)


def calculate_mt(pt, eta, phi, e, met, metphi):
    """
    Calculates the transverse mass MT and RT (closely related calcs)
    """
    met_x = np.cos(metphi) * met
    met_y = np.sin(metphi) * met
    jet_x = np.cos(phi) * pt
    jet_y = np.sin(phi) * pt
    # jet_e = np.sqrt(jets.mass2 + jets.pt**2)
    # m^2 + pT^2 = E^2 - pT^2 - pz^2 + pT^2 = E^2 - pz^2
    pz = pt * np.sinh(eta)
    transverse_e = np.sqrt(e**2 - pz**2)
    mt = np.sqrt( (transverse_e + met)**2 - (jet_x + met_x)**2 - (jet_y + met_y)**2 )
    return mt


def veto_phi_spike(eta_veto, phi_veto, eta_jets, phi_jets, rad=0.01):
    """
    Returns an array with length of the number of jets, indicating for
    each jet if it is vetoed (False) or is okay (True).
    """
    # deta is a matrix with delta_eta to every dead cell location, per jet
    # e.g. deta[0] gives you a np.array of all delta_etas to all dead cells
    deta = np.expand_dims(eta_jets, -1) - eta_veto
    assert deta.shape == ( len(eta_jets), len(eta_veto) )
    dphi = calc_dphi(np.expand_dims(phi_jets, -1), phi_veto)
    assert dphi.shape == ( len(phi_jets), len(phi_veto) )
    # Only True for jets that are not close to _any_ dead cell location
    veto_mask = np.all((deta**2 + dphi**2) > rad, axis=-1)
    assert veto_mask.shape == (len(eta_jets),)
    return veto_mask

def veto_HEM(eta,phi,pt):
    """
    rejects events with AK4 jets eta and phi within the HEM area
    only for 2018_PostHEM era
    """
    hem = lambda eta_jet,phi_jet,pt_jet : ((eta_jet<-1.4) & (eta_jet>-3) & (phi_jet<-0.87) & (phi_jet>-1.57) & (pt_jet>30))
    hemveto = np.bitwise_not(hem(eta,phi,pt))
    return hemveto

def select_jet_topology(array: ak.Array, cutflow: OrderedDict):
    # At least 2 AK15 jets
    array = array[ak.count(array['JetsAK15.fCoordinates.fPt'], axis=-1) >= 2]
    cutflow['n_ak15jets>=2'] = len(array)
    # jetid for AK15 jets
    array = array[array['JetsAK15_ID'][:,1]>0]
    cutflow['ak15jets_id'] = len(array)
    # subleading eta < 2.4 eta
    array = array[np.abs(array['JetsAK15.fCoordinates.fEta'][:,1])<2.4]
    cutflow['subl_eta<2.4'] = len(array)
    # ECF > 0
    for ecf in [
        'JetsAK15_ecfC2b1', 'JetsAK15_ecfD2b1',
        'JetsAK15_ecfM2b1', 'JetsAK15_ecfN2b2',
    ]:
        array = array[array[ecf][:,1]>0.]
    cutflow['subl_ecf>0'] = len(array)
    return array, cutflow


def select_trigger(array: ak.Array, cutflow: OrderedDict, triggers: Tuple[List[str],List[str]], trigger_branch:List[str]):
    # Triggers
    good_triggers, bad_triggers = triggers
    good_indices = np.array([trigger_branch.index(t) for t in good_triggers])
    bad_indices = np.array([trigger_branch.index(t) for t in bad_triggers])
    if len(array):
        good_decisions = array['TriggerPass'].to_numpy()[:,good_indices]
        array = array[(good_decisions == 1).any(axis=-1)]
        # only keep events that fail the bad triggers
        if len(bad_triggers):
            bad_decisions = array['TriggerPass'].to_numpy()[:,bad_indices]
            array = array[np.logical_not((bad_decisions == 1).any(axis=-1))]
    cutflow['triggers'] = len(array)
    return array, cutflow

def select_trigger_object(array: ak.Array, cutflow: OrderedDict):
    # AK8 jetpt>500
    array = array[ak.count(array['JetsAK8.fCoordinates.fPt'], axis=-1)>=1] # At least one jet
    array = array[array['JetsAK8.fCoordinates.fPt'][:,0]>500.] # leading>500
    cutflow['ak8jet.pt>500'] = len(array)
    return array, cutflow

def select_rt(array: ak.Array, cutflow: OrderedDict):
    # rtx>1.1
    rtx = np.sqrt(1. + array['MET'].to_numpy() / array['JetsAK15.fCoordinates.fPt'][:,1].to_numpy())
    array = array[rtx>1.1]
    cutflow['rtx>1.1'] = len(array)
    return array, cutflow


def select_single_muon(array: ak.Array, cutflow: OrderedDict):
    # apply preselection - muon veto + muon selection
    # (used medium ID + pt > 50 GeV + iso < 0.2 in EXO-19-020,
    #  see AN-19-061 section 4.2)
    # require the selected muon to match with the HLT muon object
    # (which should be saved in the SingleMuon ntuples) by ΔR < 0.2
    array = arrays.array
    array = array[array['NMuons']>=1]
    if len(array):
            array = array[
                (array['Muons_mediumID'][:,0])
                & (array['Muons.fCoordinates.fPt'][:,0]>50.)
                & (array['Muons_iso'][:,0]<.2)
                ]
    if len(array):
            array = array[ak.count(array['HLTMuonObjects.fCoordinates.fPt'], axis=-1) >= 1]
            array = array[calc_dr(
                array['Muons.fCoordinates.fPt'][:,0].to_numpy(),
                array['Muons.fCoordinates.fEta'][:,0].to_numpy(),
                array['HLTMuonObjects.fCoordinates.fPt'][:,0].to_numpy(),
                array['HLTMuonObjects.fCoordinates.fEta'][:,0].to_numpy(),
                ) < .2]
    cutflow['nmuons=1'] = len(array)
    array = array[array['NElectrons']==0]
    cutflow['nelectrons=0'] = len(array)
    return array, cutflow


def select_muon_veto(array: ak.Array, cutflow: OrderedDict):
    # lepton vetoes
    array = array[array['NMuons']==0]
    cutflow['nmuons=0'] = len(array)
    return array, cutflow

def select_electron_veto(array: ak.Array, cutflow:OrderedDict):
    array = array[array['NElectrons']==0]
    cutflow['nelectrons=0'] = len(array)
    return array, cutflow


def select_metdphi(array: ak.Array, cutflow: OrderedDict):
    METDphi = calc_dphi(array['JetsAK15.fCoordinates.fPhi'][:,1].to_numpy(), array['METPhi'].to_numpy())
    array = array[abs(METDphi)<1.5]
    cutflow['abs(metdphi)<1.5'] = len(array)
    return array, cutflow

def select_metfilter_standard(array: ak.Array, cutflow: OrderedDict):
    # MET filters
    # following the order from https://twiki.cern.ch/twiki/bin/viewauth/CMS/MissingETOptionalFiltersRun2#Run_2_recommendations
    met_filters = [
        'PrimaryVertexFilter',
        'globalSuperTightHalo2016Filter',
        'HBHENoiseFilter',
        'HBHEIsoNoiseFilter',
        'BadPFMuonFilter',
        'BadPFMuonDzFilter',
        'eeBadScFilter',
        'ecalBadCalibFilter',
        'hfNoisyHitsFilter'
    ]
    for f in met_filters:
        array = array[array[f]!=0] # Pass events if not 0, is that correct?
    cutflow['metfilter'] = len(array)
    return array, cutflow


def select_highpt_muon_veto(array: ak.Array, cutflow: OrderedDict):
    # muon pt < 1500 filter to avoid highMET events
    array = array[~ak.any(array['Muons.fCoordinates.fPt'] > 1500., axis=-1)]
    cutflow['muonpt<1500'] = len(array)
    return array, cutflow


def select_metfilter_custom(array: ak.Array, cutflow:OrderedDict, year: int):
    # At least 2 AK4 jets --> required for dead cell determination
    array = array[ak.count(array['Jets.fCoordinates.fPt'], axis=-1) >= 2]
    cutflow['n_ak4jets>=2'] = len(array)

    # Filter out jets that are too close to dead cells
    ak4jet_eta = array['Jets.fCoordinates.fEta'][:,1].to_numpy()
    ak4jet_phi = array['Jets.fCoordinates.fPhi'][:,1].to_numpy()
    dead_cell_mask = veto_phi_spike(
       dataqcd_eta_ecaldead[year], dataqcd_phi_ecaldead[year],
       ak4jet_eta, ak4jet_phi,
       rad = 0.01
    )

    array = array[dead_cell_mask]
    cutflow['ecaldeadcells'] = len(array)
    return array, cutflow

def select_mt(array: ak.Array, cutflow:OrderedDict):
    pt = array['JetsAK15.fCoordinates.fPt'][:,1].to_numpy()
    eta = array['JetsAK15.fCoordinates.fEta'][:,1].to_numpy()
    phi = array['JetsAK15.fCoordinates.fPhi'][:,1].to_numpy()
    e = array['JetsAK15.fCoordinates.fE'][:,1].to_numpy()

    met = array['MET'].to_numpy()
    metphi = array['METPhi'].to_numpy()
    mt = calculate_mt(pt, eta, phi, e, met, metphi)
    # compute but don't apply
    cutflow['180<mt<650'] = np.sum((mt>180) & (mt<650))
    return array, cutflow


class EventSelector():
    def __init__(self, name:str, filter_list: List[Callable]):
        self.name = name
        self.filter_list = filter_list

    def __call__(self, arrays:Arrays):
        copy = arrays.copy() # Operate on the copy
        for filter in self.filter_list:
            add_args = self._get_add_args(copy, filter)
            copy.array, copy.cutflow = filter(copy.array, copy.cutflow, **add_args)
        copy.cutflow[self.name] = len(copy.array)
        logger.debug('cutflow:\n%s', pprint.pformat(copy.cutflow))
        return copy

    @classmethod
    def _get_add_args(cls, arrays:Arrays, method: Callable):
        """Getting additional filter methods input from callable inspect method"""
        params = inspect.signature(method).parameters
        add_args = {}
        for index, name in enumerate(params):
            if index == 0:
                assert name == "array", "First argument for filter function should be array"
            elif index == 1:
                assert name == "cutflow", "Second argument should be cutflow"
            else:
                assert hasattr(arrays, name), f"Method requested parameter {name}, which does not exist for Arrays"
                add_args[name] = getattr(arrays,name)
        return add_args


filter_preselection = EventSelector("preselection", [
    select_jet_topology,
    select_trigger,
    select_trigger_object,
    select_rt,
    select_muon_veto,
    select_electron_veto,
    select_metdphi,
    select_metfilter_standard,
    select_highpt_muon_veto,
    select_metfilter_custom,
    select_mt,
])

def filter_preselection_minus_one(cut_type):
    # The selection syntax would be something like:
    # filter_preselection_minus_one("metdphi")(array)
    assert any([m.__name__ == f"select_{cut_type}" for m in filter_preselection.filter_list])
    return EventSelector(
        name=f"preselection_minus_{cut_type}",
        filter_list=[m for m in filter_preselection.filter_list if m.__name__ != f"select_{cut_type}"]
    )

def filter_hemveto(array):
    """
    Filter to cut away AK4 jets within HEM area
    only applicable for 2018 posthem era
    """
    copy = array.copy()
    a = copy.array

    eta = a['Jets.fCoordinates.fEta'][:,1].to_numpy()
    phi = a['Jets.fCoordinates.fPhi'][:,1].to_numpy()
    pt  = a['Jets.fCoordinates.fPt'][:,1].to_numpy()

    a = a[veto_HEM(eta,phi,pt)]

    copy.array = a
    copy.cut('hem_veto')
    return copy

def rhoddt_windowcuts(mt, pt, rho):
    cuts = (mt>200) & (mt<1000) & (pt>110) & (pt<1500) & (rho>-4) & (rho<0)
    return cuts

def girthmap(mt, pt,rho,girth,weight):
    from scipy.ndimage import gaussian_filter
    cuts = rhoddt_windowcuts(mt, pt, rho)
    C, RHO_edges, PT_edges = np.histogram2d(rho[cuts], pt[cuts], bins=49,weights=weight[cuts])
    w, h = 50, 50
    GIRTH_map      = [[0 for x in range(w)] for y in range(h)]
    GIRTH = girth[cuts]
    for i in range(len(RHO_edges)-1):
       for j in range(len(PT_edges)-1):
          CUT = (rho[cuts]>RHO_edges[i]) & (rho[cuts]<RHO_edges[i+1]) & (pt[cuts]>PT_edges[j]) & (pt[cuts]<PT_edges[j+1])
          if len(GIRTH[CUT])==0: continue
          if len(GIRTH[CUT])>0:
             GIRTH_map[i][j]=np.percentile(GIRTH[CUT],48)

    GIRTH_map_smooth = gaussian_filter(GIRTH_map,1)
    return GIRTH_map_smooth, RHO_edges, PT_edges


def girthddt(mt, pt,rho,girth,weight):
    cuts = rhoddt_windowcuts(mt, pt, rho)
    girth_map_smooth, RHO_edges, PT_edges = girthmap(mt, pt, rho, girth, weight)
    nbins = 49
    Pt_min, Pt_max = min(PT_edges), max(PT_edges)
    Rho_min, Rho_max = min(RHO_edges), max(RHO_edges)

    ptbin_float  = nbins*(pt-Pt_min)/(Pt_max-Pt_min)
    rhobin_float = nbins*(rho-Rho_min)/(Rho_max-Rho_min)

    ptbin  = np.clip(1 + np.round(ptbin_float).astype(int), 0, nbins)
    rhobin = np.clip(1 + np.round(rhobin_float).astype(int), 0, nbins)

    girthDDT = np.array([girth[i] - girth_map_smooth[rhobin[i]-1][ptbin[i]-1] for i in range(len(girth))])
    return girthDDT

def filter_girthDDT(array):
    copy = array.copy()
    a, cutflow = copy.array, copy.cutflow
    pt = a['JetsAK15.fCoordinates.fPt'][:,1].to_numpy()
    eta = a['JetsAK15.fCoordinates.fEta'][:,1].to_numpy()
    phi = a['JetsAK15.fCoordinates.fPhi'][:,1].to_numpy()
    e = a['JetsAK15.fCoordinates.fE'][:,1].to_numpy()

    met = a['MET'].to_numpy()
    metphi = a['METPhi'].to_numpy()
    mt = calculate_mt(
     pt, eta, phi, e,
     met, metphi
     )
    mass = calculate_mass(pt, eta, e)
    rho = calculate_rho(pt,eta,e)
    girth = a['JetsAK15_girth'][:,1].to_numpy()
    weight =a['Weight'].to_numpy()
    a['girthddt'] = girthddt(mt, pt, rho, girth,weight)

    a = a[a['girthddt']>0]

    cutflow['girthddt>0'] = len(a)
    copy.array = a
    return copy


def filter_zprime_in_cone(array):
    """
    Require both dark quarks and the zprime to be INSIDE 1.5 cone of the subleading jet
    """
    copy = array.copy()
    a, cutflow = copy.array, copy.cutflow

    # at least 1 zprime particle
    a = a[ak.sum(a['GenParticles_PdgId']==4900023, axis=-1)>=1]
    # at least 2 dark quarks
    a = a[ak.sum((np.abs(a['GenParticles_PdgId'])==4900101) & (a['GenParticles_Status']==71), axis=-1)>=2]
    cutflow['1zprime2darkquarks'] = len(a)

    # Require both dark quarks and the zprime to be within 1.5 cone of the subleading jet
    eta_subl = a['JetsAK15.fCoordinates.fEta'][:,1].to_numpy()
    phi_subl = a['JetsAK15.fCoordinates.fPhi'][:,1].to_numpy()
    eta_zprime = a['GenParticles.fCoordinates.fEta'][a['GenParticles_PdgId']==4900023][:,0].to_numpy()
    phi_zprime = a['GenParticles.fCoordinates.fPhi'][a['GenParticles_PdgId']==4900023][:,0].to_numpy()
    select_darkquarks = (np.abs(a['GenParticles_PdgId'])==4900101) & (a['GenParticles_Status']==71)
    eta_darkquark1 = a['GenParticles.fCoordinates.fEta'][select_darkquarks][:,0].to_numpy()
    phi_darkquark1 = a['GenParticles.fCoordinates.fPhi'][select_darkquarks][:,0].to_numpy()
    eta_darkquark2 = a['GenParticles.fCoordinates.fEta'][select_darkquarks][:,1].to_numpy()
    phi_darkquark2 = a['GenParticles.fCoordinates.fPhi'][select_darkquarks][:,1].to_numpy()
    a = a[
        (calc_dr(eta_subl, phi_subl, eta_zprime, phi_zprime)<=1.5)
        & (calc_dr(eta_subl, phi_subl, eta_darkquark1, phi_darkquark1)<=1.5)
        & (calc_dr(eta_subl, phi_subl, eta_darkquark2, phi_darkquark2)<=1.5)
        ]
    cutflow['zdq<1.5'] = len(a)
    copy.array = a
    return copy


def filter_zprime_not_in_cone(array):
    """
    Require both dark quarks and the zprime to be OUTSIDE 1.5 cone of the subleading jet
    """
    copy = array.copy()
    a, cutflow = copy.array, copy.cutflow

    # at least 1 zprime particle
    a = a[ak.sum(a['GenParticles_PdgId']==4900023, axis=-1)>=1]
    # at least 2 dark quarks
    a = a[ak.sum((np.abs(a['GenParticles_PdgId'])==4900101) & (a['GenParticles_Status']==71), axis=-1)>=2]
    cutflow['1zprime2darkquarks'] = len(a)

    # Require both dark quarks and the zprime to be OUTSIDE 1.5 cone of the subleading jet
    eta_subl = a['JetsAK15.fCoordinates.fEta'][:,1].to_numpy()
    phi_subl = a['JetsAK15.fCoordinates.fPhi'][:,1].to_numpy()
    eta_zprime = a['GenParticles.fCoordinates.fEta'][a['GenParticles_PdgId']==4900023][:,0].to_numpy()
    phi_zprime = a['GenParticles.fCoordinates.fPhi'][a['GenParticles_PdgId']==4900023][:,0].to_numpy()
    select_darkquarks = (np.abs(a['GenParticles_PdgId'])==4900101) & (a['GenParticles_Status']==71)
    eta_darkquark1 = a['GenParticles.fCoordinates.fEta'][select_darkquarks][:,0].to_numpy()
    phi_darkquark1 = a['GenParticles.fCoordinates.fPhi'][select_darkquarks][:,0].to_numpy()
    eta_darkquark2 = a['GenParticles.fCoordinates.fEta'][select_darkquarks][:,1].to_numpy()
    phi_darkquark2 = a['GenParticles.fCoordinates.fPhi'][select_darkquarks][:,1].to_numpy()
    a = a[
        (calc_dr(eta_subl, phi_subl, eta_zprime, phi_zprime)>1.5)
        & (calc_dr(eta_subl, phi_subl, eta_darkquark1, phi_darkquark1)>1.5)
        & (calc_dr(eta_subl, phi_subl, eta_darkquark2, phi_darkquark2)>1.5)
        ]

    cutflow['zdq>1.5'] = len(a)
    copy.array = a
    return copy


def filter_zprime_half_in_cone(array):
    """
    Require at least 1 dark quark / zprime to be OUTSIDE 1.5 cone around subjet,
    AND     at least 1 dark quark / zprime to be INSIDE 1.5 cone around subjet
    """
    copy = array.copy()
    a, cutflow = copy.array, copy.cutflow

    # at least 1 zprime particle
    a = a[ak.sum(a['GenParticles_PdgId']==4900023, axis=-1)>=1]
    # at least 2 dark quarks
    a = a[ak.sum((np.abs(a['GenParticles_PdgId'])==4900101) & (a['GenParticles_Status']==71), axis=-1)>=2]
    cutflow['1zprime2darkquarks'] = len(a)

    # Require at least 1 dark quark / zprime to be OUTSIDE 1.5 cone around subjet,
    # AND     at least 1 dark quark / zprime to be INSIDE 1.5 cone around subjet
    eta_subl = a['JetsAK15.fCoordinates.fEta'][:,1].to_numpy()
    phi_subl = a['JetsAK15.fCoordinates.fPhi'][:,1].to_numpy()
    eta_zprime = a['GenParticles.fCoordinates.fEta'][a['GenParticles_PdgId']==4900023][:,0].to_numpy()
    phi_zprime = a['GenParticles.fCoordinates.fPhi'][a['GenParticles_PdgId']==4900023][:,0].to_numpy()
    select_darkquarks = (np.abs(a['GenParticles_PdgId'])==4900101) & (a['GenParticles_Status']==71)
    eta_darkquark1 = a['GenParticles.fCoordinates.fEta'][select_darkquarks][:,0].to_numpy()
    phi_darkquark1 = a['GenParticles.fCoordinates.fPhi'][select_darkquarks][:,0].to_numpy()
    eta_darkquark2 = a['GenParticles.fCoordinates.fEta'][select_darkquarks][:,1].to_numpy()
    phi_darkquark2 = a['GenParticles.fCoordinates.fPhi'][select_darkquarks][:,1].to_numpy()
    a = a[
        ((calc_dr(eta_subl, phi_subl, eta_zprime, phi_zprime)<=1.5)
        | (calc_dr(eta_subl, phi_subl, eta_darkquark1, phi_darkquark1)<=1.5)
        | (calc_dr(eta_subl, phi_subl, eta_darkquark2, phi_darkquark2)<=1.5))
        & ((calc_dr(eta_subl, phi_subl, eta_zprime, phi_zprime)>1.5)
        | (calc_dr(eta_subl, phi_subl, eta_darkquark1, phi_darkquark1)>1.5)
        | (calc_dr(eta_subl, phi_subl, eta_darkquark2, phi_darkquark2)>1.5))
        ]

    cutflow['half_truth'] = len(a)
    copy.array = a
    return copy


def filter_stitch(array):
    """
    Filter to cut away kinematic regions from ttjets/wjets that are covered in
    dedicated samples
    """
    copy = array.copy()
    a = copy.array

    if array.bkg_type == 'ttjets':
        if 'htbin' in array.metadata:
            # HT bin
            a = a[a['madHT']>=600.]
        elif 'n_lepton_sample' in array.metadata:
            # SingleLep or DiLep
            if array.metadata.get('genmet_sample', False):
                a = a[(a['madHT']<600.) & (a['GenMET']>=150.)]
            else:
                a = a[(a['madHT']<600.) & (a['GenMET']<150.)]
        else:
            #correction for number of leptons
            n_leptons = (
                ak.count(a['GenElectrons.fCoordinates.fPt'], axis=-1)
                + ak.count(a['GenMuons.fCoordinates.fPt'], axis=-1)
                + ak.count(a['GenTaus.fCoordinates.fPt'], axis=-1)
                )
            a = a[(a['madHT']<600.) & (n_leptons==0)]

    elif array.bkg_type == 'wjets':
        if 'htbin' in array.metadata:
            a = a[a['madHT']>100.]
        else:
            # Inclusive
            a = a[a['madHT']<=100.]

    copy.array = a
    copy.cut('stitch')
    return copy


def filter_at_least_one_ak8jet(array):
    return filter_at_least_n_jets(array, n=1, cone=8)


def filter_at_least_n_jets(array, n=1, cone=8):
    """
    Returns an Array that has at least n jets in the ak-collection.
    `ak` can be 4, 8, or 15.
    """
    if cone in [8, 15]:
        cone = 'AK' + str(cone)
    elif cone == 4:
        cone = ''
    else:
        raise Exception('Parameter cone should 4, 8, or 15')
    njets = ak.count(array.array['Jets{}.fCoordinates.fPt'.format(cone)], axis=-1)
    copy = array.copy()
    copy.array = copy.array[njets>=n]
    copy.cutflow['>={}{}jets'.format(n, cone)] = len(copy.array)
    return copy

def filter_at_least_n_muons(array, n=1):
    nmuons = ak.count(array.array['Muons.fCoordinates.fPt'], axis=-1)
    copy = array.copy()
    copy.array = copy.array[nmuons>=n]
    copy.cutflow['>={}muons'.format(n)] = len(copy.array)
    return copy


class Columns:
    """
    Class that contains a dictionary of arrays. Each array is guaranteed
    to be shaped (n_events,).

    This class is designed for read/write to disk.
    """
    @classmethod
    def load(cls, infile, encoding='ASCII'):
        try:
            with local_copy(infile) as local:
                d = np.load(local, allow_pickle=True, encoding=encoding)
        except:
            logger.error('Error opening %s', infile)
            raise
        inst = cls()
        inst.arrays = d['arrays'].item()
        inst.metadata = d['metadata'].item()
        inst.metadata['src'] = infile
        for key, val in zip(d['cutflow_keys'], d['cutflow_vals']):
            inst.cutflow[key] = val
        return inst

    def __init__(self):
        self.arrays = {}
        self.metadata = {}
        self.cutflow = OrderedDict()
        self._xs = None

    def __len__(self):
        for v in self.arrays.values():
            return len(v)

    def __repr__(self):
        return '<Columns {0}>'.format(pprint.pformat(self.metadata))

    def select(self, where):
        the_copy = self.copy()
        the_copy.arrays = {k: v[where] for k, v in self.arrays.items()}
        return the_copy

    def save(self, outfile, force=False):
        import seutils
        do_stageout = False
        if seutils.path.has_protocol(outfile):
            remote_outfile = outfile
            outfile = uid() + '.npz'
            do_stageout = True

        cutflow_keys = []
        cutflow_vals = []
        for key, val in self.cutflow.items():
            cutflow_keys.append(key)
            cutflow_vals.append(val)
        cutflow_vals = np.array(cutflow_vals)

        logger.info('Dumping to %s', outfile)

        # Automatically create parent directory if not existent
        outdir = osp.dirname(osp.abspath(outfile))
        if not osp.isdir(outdir):
            os.makedirs(outdir)

        np.savez(
            outfile,
            arrays = self.arrays,
            metadata = dict(self.metadata, svj_ntuple_processing_version=version()),
            cutflow_keys = cutflow_keys,
            cutflow_vals = cutflow_vals
            )

        if do_stageout:
            logger.info('Staging out %s -> %s', outfile, remote_outfile)
            seutils.cp(outfile, remote_outfile, force=force)
            os.remove(outfile)

    def to_numpy(self, features=None):
        """
        Returns the various arrays as a rectangular numpy array.
        If `features` is None, all features are used, sorted alphabetically.
        """
        if features is None: features = list(sorted(self.arrays.keys()))
        # Check if passed features exist in this .npz
        missing_features = []
        for f in features:
            if f not in self.arrays.keys():
                logger.error('Feature %s is not available in %s.', f, self)
                missing_features.append(f)
        if missing_features:
            raise Exception(
                'Cannot build numpy array.'
                ' Available features: %s;'
                ' Missing requested features: %s'
                % list(self.arrays.keys()), missing_features
                )
        X = []
        for f in features:
            X.append(self.arrays[f])
        X = np.column_stack(X)
        return X

    def to_dataframe(self, features=None):
        """
        Returns the various arrays as a pandas.DataFrame. The self.arrays keys are used
        as column names
        """
        import pandas as pd
        return pd.DataFrame.from_dict(self.arrays)

    def copy(self):
        copy = self.__class__()
        copy.metadata = self.metadata.copy()
        copy.arrays = self.arrays.copy()
        copy.cutflow = self.cutflow.copy()
        return copy

    @property
    def xs(self):
        if 'bkg_type' in self.metadata:
            raise NotImplementedError('XS not gettable for background')
        if self._xs is None:
            # Load the signal cross section polynomial
            import requests
            fit = np.poly1d(
                requests
                .get('https://raw.githubusercontent.com/boostedsvj/svj_madpt_crosssection/main/fit_madpt300.txt')
                .json()
                )
            self._xs = fit(self.metadata['mz'])
        return self._xs


def load_numpy(infile, features):
    """
    Convenience function
    """
    return Columns.load(infile).to_numpy(features)


def concat_columns(columns):
    """
    Concatenates columns into one column object.
    Metadata is taken from the first column.
    Keys are taken from first column, and are assumed to exist in others!
    """
    cols = Columns()
    cols.metadata = columns[0].metadata.copy()

    # Concatenated arrays; 1 call per key
    for key in columns[0].arrays.keys():
        try:
            cols.arrays[key] = np.concatenate([c.arrays[key] for c in columns])
        except KeyError:
            # Find the column that crashed:
            for c in columns:
                if key not in c.arrays:
                    logger.error(
                        'Key %s does not exist in columns %s;'
                        ' expected columns for concatenation: %s;'
                        ' available columns: {list(c.arrays.keys())}.',
                        key, c, list(columns[0].arrays.keys())
                        )
            raise

    # Summed cutflow
    for key in columns[0].cutflow.keys():
        cols.cutflow[key] = sum(c.cutflow[key] for c in columns)

    return cols


def bdt_feature_columns(array, load_mc=False, save_scale_weights=False):
    """
    Takes an Array object, calculates needed columns for the bdt training.
    """
    cols = Columns()
    cols.metadata = array.metadata.copy()
    cols.cutflow = array.cutflow.copy()
    # Prepare features
    arr = array.array
    a = {}
    a['girth'] = arr['JetsAK15_girth'][:,1].to_numpy()
    a['ptd'] = arr['JetsAK15_ptD'][:,1].to_numpy()
    a['axismajor'] = arr['JetsAK15_axismajor'][:,1].to_numpy()
    a['axisminor'] = arr['JetsAK15_axisminor'][:,1].to_numpy()
    a['ecfm2b1'] = arr['JetsAK15_ecfM2b1'][:,1].to_numpy()
    a['ecfm2b2'] = arr['JetsAK15_ecfM2b2'][:,1].to_numpy()
    a['ecfd2b1'] = arr['JetsAK15_ecfD2b1'][:,1].to_numpy()
    a['ecfd2b2'] = arr['JetsAK15_ecfD2b2'][:,1].to_numpy()
    a['ecfc2b1'] = arr['JetsAK15_ecfC2b1'][:,1].to_numpy()
    a['ecfc2b2'] = arr['JetsAK15_ecfC2b2'][:,1].to_numpy()
    a['ecfn2b1'] = arr['JetsAK15_ecfN2b1'][:,1].to_numpy()
    a['ecfn2b2'] = arr['JetsAK15_ecfN2b2'][:,1].to_numpy()
    a['metdphi'] = calc_dphi(arr['JetsAK15.fCoordinates.fPhi'][:,1].to_numpy(), arr['METPhi'].to_numpy())

    if load_mc: a['weight'] = arr['Weight'].to_numpy()
    a['met'] = arr['MET'].to_numpy()
    a['metphi'] = arr['METPhi'].to_numpy()

    # Save some extra vars for potential reweighting / other analysis
    a['pt'] = arr['JetsAK15.fCoordinates.fPt'][:,1].to_numpy()
    a['eta'] = arr['JetsAK15.fCoordinates.fEta'][:,1].to_numpy()
    a['phi'] = arr['JetsAK15.fCoordinates.fPhi'][:,1].to_numpy()
    a['e'] = arr['JetsAK15.fCoordinates.fE'][:,1].to_numpy()
    a['mass'] = calculate_mass(a['pt'], a['eta'], a['e'])
    a['mt'] = calculate_mt(
        a['pt'], a['eta'], a['phi'], a['e'],
        a['met'], a['metphi']
        )
    a['run'] = arr['RunNum'].to_numpy()
    a['lumiblock'] = arr['LumiBlockNum'].to_numpy()
    a['evt'] = arr['EvtNum'].to_numpy()

    a['ak15_chad_ef'] = arr['JetsAK15_chargedHadronEnergyFraction'][:,1].to_numpy()
    a['ak15_nhad_ef'] = arr['JetsAK15_neutralHadronEnergyFraction'][:,1].to_numpy()
    a['ak15_elect_ef'] = arr['JetsAK15_electronEnergyFraction'][:,1].to_numpy()
    a['ak15_muon_ef'] = arr['JetsAK15_muonEnergyFraction'][:,1].to_numpy()
    a['ak15_photon_ef'] = arr['JetsAK15_photonEnergyFraction'][:,1].to_numpy()

    a['rho'] = calculate_rho(a['pt'], a['eta'], a['e'])
    #a['girthddt'] = girthddt(a['mt'], a['pt'],a['rho'],a['girth'],a['weight'])
    a['rt'] = calc_rt(a['met'], a['pt'])

    a['leading_pt'] = arr['JetsAK15.fCoordinates.fPt'][:,0].to_numpy()
    a['leading_eta'] = arr['JetsAK15.fCoordinates.fEta'][:,0].to_numpy()
    a['leading_phi'] = arr['JetsAK15.fCoordinates.fPhi'][:,0].to_numpy()
    a['leading_e'] = arr['JetsAK15.fCoordinates.fE'][:,0].to_numpy()
    a['leading_mass'] = calculate_mass(a['leading_pt'], a['leading_eta'], a['leading_e'])
    a['leading_mt'] = calculate_mt(
        a['leading_pt'], a['leading_eta'], a['leading_phi'], a['leading_e'],
        a['met'], a['metphi']
        )

    a['ak4_lead_eta'] = arr['Jets.fCoordinates.fEta'][:,0].to_numpy()
    a['ak4_lead_phi'] = arr['Jets.fCoordinates.fPhi'][:,0].to_numpy()
    a['ak4_lead_pt'] = arr['Jets.fCoordinates.fPt'][:,0].to_numpy()
    a['ak4_subl_eta'] = arr['Jets.fCoordinates.fEta'][:,1].to_numpy()
    a['ak4_subl_phi'] = arr['Jets.fCoordinates.fPhi'][:,1].to_numpy()
    a['ak4_subl_pt'] = arr['Jets.fCoordinates.fPt'][:,1].to_numpy()
    a['ak8_lead_pt'] = ak.fill_none(ak.firsts(arr['JetsAK8.fCoordinates.fPt']), -1).to_numpy()
    a['ak8_lead_phi']= ak.fill_none(ak.firsts(arr['JetsAK8.fCoordinates.fPhi']), -1).to_numpy()
    a['ak8_lead_eta']= ak.fill_none(ak.firsts(arr['JetsAK8.fCoordinates.fEta']), -1).to_numpy()

    a['jetsak15_id'] = arr['JetsAK15_ID'][:,1].to_numpy()
    if load_mc:
        a['puweight'] = arr['puWeight'].to_numpy()

    if save_scale_weights:
        a['scaleweights'] = arr['ScaleWeights'].to_numpy()

    # QCD high MET events
    a['lead_muonpt']    = ak.fill_none(ak.firsts(arr['Muons.fCoordinates.fPt']), -1.).to_numpy()
    a['nmuons']         = arr['NMuons'].to_numpy()
    a['nelectrons']     = arr['NElectrons'].to_numpy()

    # add ecf with full jet components
    a['ecfm2b1full'] = arr['JetsAK15_ecfFullM2b1'][:,1].to_numpy()
    a['ecfm2b2full'] = arr['JetsAK15_ecfFullM2b2'][:,1].to_numpy()
    a['ecfn2b1full'] = arr['JetsAK15_ecfFullN2b1'][:,1].to_numpy()
    a['ecfn2b2full'] = arr['JetsAK15_ecfFullN2b2'][:,1].to_numpy()
    a['ecfc2b1full'] = arr['JetsAK15_ecfFullC2b1'][:,1].to_numpy()
    a['ecfc2b2full'] = arr['JetsAK15_ecfFullC2b2'][:,1].to_numpy()
    a['ecfd2b1full'] = arr['JetsAK15_ecfFullD2b1'][:,1].to_numpy()
    a['ecfd2b2full'] = arr['JetsAK15_ecfFullD2b2'][:,1].to_numpy()

    # more branches
    a['JetsAK15_nConstituents']         = arr['JetsAK15_nConstituents'][:,1].to_numpy()
    a['JetsAK15_nConstituentsSoftDrop'] = arr['JetsAK15_nConstituentsSoftDrop'][:,1].to_numpy()

    cols.arrays = a
    return cols

# Backward compatibility
cr_feature_columns = bdt_feature_columns

def nminus_one_columns(array, skip_cut, load_mc=False):
    """
    Takes an Array object, calculates needed columns for the bdt training.
    """
    cols = Columns()
    cols.metadata = array.metadata.copy()
    cols.cutflow = array.cutflow.copy()
    # Prepare features
    arr = array.array
    a = {}

    # Always save event ID
    a['run'] = arr['RunNum'].to_numpy()
    a['lumiblock'] = arr['LumiBlockNum'].to_numpy()
    a['evt'] = arr['EvtNum'].to_numpy()
    # And the main analysis variable
    a['mt'] = calculate_mt(
        arr['JetsAK15.fCoordinates.fPt'][:, 1].to_numpy(),
        arr['JetsAK15.fCoordinates.fEta'][:, 1].to_numpy(),
        arr['JetsAK15.fCoordinates.fPhi'][:, 1].to_numpy(),
        arr['JetsAK15.fCoordinates.fE'][:, 1].to_numpy(),
        arr['MET'].to_numpy(),
        arr['METPhi'].to_numpy(),
        )
    # And event weights
    if load_mc:
        a['weight'] = arr['Weight'].to_numpy()
        a['puweight'] = arr['puWeight'].to_numpy()

    # For everything else, we should only include the variable that was skipped
    if skip_cut == 'metdphi':
        a['metdphi'] = calc_dphi(arr['JetsAK15.fCoordinates.fPhi'][:,1].to_numpy(), arr['METPhi'].to_numpy())
    elif skip_cut == 'rt':
        a['met'] = arr['MET'].to_numpy()
        a['rt'] = calc_rt(a['met'], arr['JetsAK15.fCoordinates.fPt'][:,1].to_numpy())
    elif skip_cut == "muon_veto":
        a['nmuons'] = arr['NMuons'].to_numpy()
    elif skip_cut == "electron_veto":
        a['nelectrons'] = arr['NElectrons'].to_numpy()

    cols.arrays = a
    return cols

def triggerstudy_columns(array, is_mc=True, single_muon_trigs=False, all_triggers=False):
    a = array.array # Just to avoid typing array.array everywhere

    trigger_names = array.trigger_branch

    if all_triggers:
        trigger_set = set(trigger_names)
        keep_trigger_mask = np.ones(len(trigger_names), dtype=bool)
    else:
        # Most triggers are not interesting for us and they take up space
        # Select all the triggers we would like to keep for further study
        trigger_set = set(triggers_2016 + triggers_2017 + triggers_2018)
        if single_muon_trigs:
            trigger_set.update({
                'HLT_Mu50_IsoVVVL_PFHT400_v',
                'HLT_Mu50_IsoVVVL_PFHT450_v',
                'HLT_Mu50_v',
                'HLT_TkMu50_v',
                })
        # Save some more to be sure - not sure if needed
        trigger_set.update({
            'HLT_PFMETTypeOne120_PFMHT120_IDTight_v',  # 2017 MET trigger?
            'HLT_PFMET120_PFMHT120_IDTight_v',         # 2018 MET trigger?
            'HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_v', # 2018 MET trigger?
            })
        keep_trigger_mask = np.array([(t in trigger_set) for t in trigger_names])

    cols = Columns()
    cols.cutflow = array.cutflow
    cols.metadata = array.metadata
    cols.metadata['trigger_titles'] = [t for t in array.trigger_branch if t in trigger_set]

    cols.arrays['triggers'] = a['TriggerPass'][:, keep_trigger_mask].to_numpy()
    assert cols.arrays['triggers'].shape[1] == len(cols.metadata['trigger_titles'])

    # Event-level variables
    cols.arrays['ht'] = a['HT'].to_numpy()
    cols.arrays['met'] = a['MET'].to_numpy()
    if is_mc: cols.arrays['weight'] = a['Weight'].to_numpy()

    # AK8 jets
    pt_ak8 = a['JetsAK8.fCoordinates.fPt']
    njets = ak.count(pt_ak8, axis=-1).to_numpy()
    cols.arrays['njets'] = njets

    cols.arrays['pt'] = np.ones_like(njets) * np.nan
    if np.any(njets>=1): cols.arrays['pt'][njets>=1] = pt_ak8[njets>=1,0].to_numpy()
    cols.arrays['pt_subl'] = np.ones_like(cols.arrays['pt']) * np.nan
    if np.any(njets>=2): cols.arrays['pt_subl'][njets>=2] = pt_ak8[njets>=2,1].to_numpy()

    # AK15 jets - tricky, because most events won't have a subl AK15 jet
    njets_ak15 = ak.count(a['JetsAK15.fCoordinates.fPt'], axis=-1)
    cols.arrays['njets_ak15'] = njets_ak15

    # Leading
    pt_ak15_lead = a['JetsAK15.fCoordinates.fPt'][(njets_ak15>=1),0].to_numpy()
    cols.arrays['pt_ak15'] = np.ones_like(cols.arrays['pt']) * np.nan
    cols.arrays['pt_ak15'][njets_ak15>=1] = pt_ak15_lead

    # Subleading
    # Filter array: Get events with at least 2 AK15 jets, calculate MT_subl
    array_ak15 = filter_at_least_n_jets(array, n=2, cone=15)
    pt_ak15_subl = array_ak15.array['JetsAK15.fCoordinates.fPt'][:,1].to_numpy()
    mt_ak15_subl = calculate_mt(
        pt_ak15_subl,
        array_ak15.array['JetsAK15.fCoordinates.fEta'][:,1].to_numpy(),
        array_ak15.array['JetsAK15.fCoordinates.fPhi'][:,1].to_numpy(),
        array_ak15.array['JetsAK15.fCoordinates.fE'][:,1].to_numpy(),
        array_ak15.array['MET'].to_numpy(), array_ak15.array['METPhi'].to_numpy()
        )
    # Use NaN where there was no subl ak15 jet
    cols.arrays['mt_ak15_subl'] = np.ones_like(cols.arrays['pt']) * np.nan
    cols.arrays['mt_ak15_subl'][njets_ak15>=2] = mt_ak15_subl
    # Store pT while we're at it
    cols.arrays['pt_ak15_subl'] = np.ones_like(cols.arrays['pt']) * np.nan
    cols.arrays['pt_ak15_subl'][njets_ak15>=2] = pt_ak15_subl
    return cols

from .systematics import *
