"""Module to create valid beam spectra for a given set of instruments."""

# Load files and do flood correction
# import mantid algorithms, numpy and matplotlib
from mantid.simpleapi import *


def pre_process(runno, detectors='75-95'):
    """Pre-process data"""
    LoadISISNexus(
        str(runno), OutputWorkspace=str(runno), LoadMonitors='Exclude'
    )
    ConvertUnits(
        InputWorkspace=str(runno),
        OutputWorkspace=str(runno),
        Target='Wavelength',
        AlignBins=True,
    )
    try:
        SumSpectra(
            InputWorkspace=str(runno),
            OutputWorkspace=str(runno),
            ListOfWorkspaceIndices=detectors,
        )
    except BaseException:
        ExtractSpectra(
            InputWorkspace=str(runno),
            OutputWorkspace=str(runno),
            WorkspaceIndexList=detectors,
        )
    ReplaceSpecialValues(
        InputWorkspace=str(runno),
        OutputWorkspace=str(runno),
        NaNValue=0,
        InfinityValue=0,
    )
    NormaliseByCurrent(str(runno), OutputWorkspace=str(runno))
    CropWorkspace(str(runno), XMin=1.0, XMax=17.0, OutputWorkspace=str(runno))
    Rebin(str(runno), Params=0.005, OutputWorkspace=str(runno))


# INTER
pre_process(62096)  # Both DB's to get correct flux 0.5 degrees
pre_process(62097)
Scale(
    '62096', Factor=13, OutputWorkspace='62096'
)  # Quickest way is to do this by hand I think
a = 0.7 / 0.5
Scale(
    '62096', Factor=a, OutputWorkspace='62096'
)  # To make it effectively at 0.7 degrees, same as OFFSPEC beam
SaveAscii(
    '62096',
    Filename=('/mnt/ceph/home/jc15575/Desktop/Enough/enough-is-enough-main'
              '/simulation/INTER_Air_DB.dat'),
    WriteSpectrumID=False,
    ColumnHeader=False,
)

# POLREF
pre_process(34365, detectors='272-285')  # 0.5 degrees I think
a = 0.7 / 0.5
Scale(
    '34365', Factor=a, OutputWorkspace='34365'
)  # To make it effectively at 0.7 degrees, same as OFFSPEC beam
SaveAscii(
    '34365',
    Filename=('/mnt/ceph/home/jc15575/Desktop/Enough/enough-is-enough-main/'
              'simulation/POLREF_Air_DB.dat'),
    WriteSpectrumID=False,
    ColumnHeader=False,
)

# OFFSPEC
pre_process(
    62802, detectors='380-420'
)  # Both DB's to get correct flux, 0.7 degrees
pre_process(62803, detectors='380-420')
Scale(
    '62802', Factor=3.7, OutputWorkspace='62802'
)  # Quickest way is to do this by hand I think
SaveAscii(
    '62802',
    Filename=('/mnt/ceph/home/jc15575/Desktop/Enough/enough-is-enough-main'
              '/simulation/OFFSPEC_Air_DB.dat'),
    WriteSpectrumID=False,
    ColumnHeader=False,
)
pre_process(53439, detectors='380-420')  # Polarised

# SURF
pre_process(137344, detectors=0)  # 0.35 degrees I think
a = 0.7 / 0.35
Scale('137344', Factor=a, OutputWorkspace='137344')
SaveAscii(
    '137344',
    Filename=('/mnt/ceph/home/jc15575/Desktop/Enough/enough-is-enough-main'
              '/simulation/SURF_Air_DB.dat'),
    WriteSpectrumID=False,
    ColumnHeader=False,
)


LoadNexusProcessed(
    Filename=r'C:/Users/npi34092/Dropbox/Offspec_reduction\GC_Flood_2-19.nxs',
    OutputWorkspace='wsf1b_flood',
)
ConvertUnits(
    InputWorkspace='wsf1b_flood',
    OutputWorkspace='wsf1b_flood',
    Target='Wavelength',
    AlignBins=True,
)
LoadISISNexus(
    Filename=(r'\\isis.cclrc.ac.uk\inst$\ndxoffspec\instrument\data'
              r'\cycle_18_4\OFFSPEC00050612.nxs'),
    OutputWorkspace='wtemp',
)
CloneWorkspace(InputWorkspace='wtemp', OutputWorkspace='w55')
LoadISISNexus(
    Filename=(r'\\isis.cclrc.ac.uk\inst$\ndxoffspec\instrument\data'
              r'\cycle_18_4\OFFSPEC00050613.nxs'),
    OutputWorkspace='wtemp',
)
CloneWorkspace(InputWorkspace='wtemp', OutputWorkspace='w56')
ConvertUnits(
    InputWorkspace='w55',
    OutputWorkspace='w55',
    Target='Wavelength',
    AlignBins=True,
)
CropWorkspace(
    InputWorkspace='w55',
    OutputWorkspace='w55det',
    StartWorkspaceIndex=5,
    EndWorkspaceIndex=771,
)
RebinToWorkspace(
    WorkspaceToRebin='wsf1b_flood',
    WorkspaceToMatch='w55det',
    OutputWorkspace='wsf1b_flood_reb',
)
Divide(
    LHSWorkspace='w55det',
    RHSWorkspace='wsf1b_flood_reb',
    OutputWorkspace='w55det',
    AllowDifferentNumberSpectra=True,
)

# Sum up the direct beam spectra and do the same for the second workspace
SumSpectra(
    InputWorkspace='w55det',
    OutputWorkspace='w55norm',
    ListOfWorkspaceIndices='384-414',
)
ReplaceSpecialValues(
    InputWorkspace='w55norm',
    OutputWorkspace='w55norm',
    NaNValue=0,
    InfinityValue=0,
)
ConvertUnits(
    InputWorkspace='w56',
    OutputWorkspace='w56',
    Target='Wavelength',
    AlignBins=True,
)
CropWorkspace(
    InputWorkspace='w56',
    OutputWorkspace='w56det',
    StartWorkspaceIndex=5,
    EndWorkspaceIndex=771,
)
RebinToWorkspace(
    WorkspaceToRebin='wsf1b_flood',
    WorkspaceToMatch='w56det',
    OutputWorkspace='wsf1b_flood_reb',
)
Divide(
    LHSWorkspace='w56det',
    RHSWorkspace='wsf1b_flood_reb',
    OutputWorkspace='w56det',
    AllowDifferentNumberSpectra=True,
)
SumSpectra(
    InputWorkspace='w56det',
    OutputWorkspace='w56norm',
    ListOfWorkspaceIndices='384-414',
)
ReplaceSpecialValues(
    InputWorkspace='w56norm',
    OutputWorkspace='w56norm',
    NaNValue=0,
    InfinityValue=0,
)


# For a single direct beam I think this will do. s1/s2(h) = 1.95 (30) /
# 0.1725 (30)
NormaliseByCurrent('w56norm', OutputWorkspace='w56norm')
CropWorkspace('w56norm', XMin=1.0, XMax=14.0, OutputWorkspace='w56norm')
Rebin('w56norm', Params=0.05, OutputWorkspace='w56norm')
"""
#Scale the direct beam with smaller slits correctly and add them together
Integration(InputWorkspace='w55norm', OutputWorkspace='w55int', RangeLower=1,
            RangeUpper=14)
Integration(InputWorkspace='w56norm', OutputWorkspace='w56int', RangeLower=1,
            RangeUpper=14)
Multiply(LHSWorkspace='w55norm', RHSWorkspace='w56int',
         OutputWorkspace='w55norm')
Divide(LHSWorkspace='w55norm', RHSWorkspace='w55int',
       OutputWorkspace='w55norm')
MultiplyRange(InputWorkspace='w56norm', OutputWorkspace='w56norm', EndBin=157)
WeightedMean(InputWorkspace1='w55norm', InputWorkspace2='w56norm',
             OutputWorkspace='DBair03')

# import mantid algorithms
from mantid.simpleapi import *

"""
