import svj_ntuple_processing as svj
import numpy as np

variables = np.array(['ak8_lead_pt', 'girth', 'ptd', 'axismajor', 'axisminor', 'ecfm2b1', 'ecfd2b1', 'ecfc2b1', 'ecfn2b2', 'metdphi', 'weight', 'met', 'metphi', 'pt', 'eta', 'phi', 'e', 'mt', 'rt', 'leading_pt', 'leading_eta', 'leading_phi', 'leading_e', 'ak4_lead_eta', 'ak4_lead_phi', 'ak4_lead_pt', 'ak4_subl_eta', 'ak4_subl_phi', 'ak4_subl_pt', 'EcalDeadCellTriggerPrimitiveFilter', 'EcalDeadCellBoundaryEnergyFilter'])

#mz = np.array(['150', '250', '350', '450', '550'])
#mz = np.array(['200', '250', '350', '450', '550'])
mz = np.array(['250', '300', '350', '400', '450', '500', '550'])
#rinv = np.array(['0.1', '0.3', '0.7'])
rinv = np.array(['0.3'])
#mdark = np.array(['5'])

#mz = np.array(['450', '550'])
#rinv = np.array(['0.3'])
#mdark = np.array(['10'])

fermilab_files = 'root://cmseos.fnal.gov//store/user/lpcdarkqcd/boosted/orthogonalitystudy/HADD/madpt300_mz'
old_fermilab_files = 'root://cmseos.fnal.gov//store/user/lpcdarkqcd/boosted/ntuples_Summer21_Sept28_hadd/TREEMAKER_genjetpt375_mZprime'
umd_files = '/home/snabili/hadoop/SIG/UL_mdark5/TREEMAKER/HADD/madpt300_mz'

signal = {}
signal_truth = {}
signal_not_truth = {}
signal_half_truth = {}
#sig_columns    = {}
for i in range(len(mz)):
   for j in range(len(rinv)):
      arrays = svj.open_root(old_fermilab_files+str(mz[i])+'_rinv'+str(rinv[j])+'.root')
      arrays = svj.cr_filter_preselection(arrays)
      #arrays, arrays_nt = svj.filter_zprime_in_cone(arrays)
      truth = svj.filter_zprime_in_cone(arrays)
      columns_truth = svj.bdt_feature_columns(truth)
      not_truth = svj.filter_zprime_not_in_cone(arrays)
      columns_not_truth = svj.bdt_feature_columns(not_truth)
      columns = svj.bdt_feature_columns(arrays)
      half_truth = svj.filter_zprime_half_in_cone(arrays)
      columns_half_truth = svj.bdt_feature_columns(half_truth)

      for k in range(len(variables)):
         #signal[variables[k]] = columns.arrays[variables[k]]
         signal_truth[variables[k]] = columns_truth.arrays[variables[k]]
         signal_not_truth[variables[k]] = columns_not_truth.arrays[variables[k]]
         signal[variables[k]] = columns.arrays[variables[k]]
         signal_half_truth[variables[k]] = columns_half_truth.arrays[variables[k]]
      #np.savez('signal_files/mz_'+mz[i]+'_rinv_'+rinv[j]+'.npz', **signal)
      np.savez('signal_files/truth/old_sigfiles/old_mz_'+mz[i]+'_rinv_'+rinv[j]+'.npz', **signal)
      np.savez('signal_files/truth/old_sigfiles/old_mz_'+mz[i]+'_rinv_'+rinv[j]+'_truth.npz', **signal_truth)
      np.savez('signal_files/truth/old_sigfiles/old_mz_'+mz[i]+'_rinv_'+rinv[j]+'_not_truth.npz', **signal_not_truth)
      np.savez('signal_files/truth/old_sigfiles/old_mz_'+mz[i]+'_rinv_'+rinv[j]+'_half_truth.npz', **signal_half_truth)


      '''print(fermilab_files+str(mz[i])+'_mdark10_rinv'+str(rinv[j])+'.root')
      print('mz_'+mz[i]+'_rinv_'+rinv[j])
      arrays = svj.open_root(fermilab_files+str(mz[i])+'_mdark10_rinv'+str(rinv[j])+'.root')
      arrays = svj.cr_filter_preselection(arrays)
      columns = svj.bdt_feature_columns(arrays)


      arrays = svj.open_root(fermilab_files+str(mz[i])+'_mdark10_rinv'+str(rinv[j])+'.root')
      arrays = svj.cr_filter_preselection(arrays)
      #arrays, arrays_nt = svj.filter_zprime_in_cone(arrays)
      truth = svj.filter_zprime_in_cone(arrays)
      columns_truth = svj.bdt_feature_columns(truth)
      not_truth = svj.filter_zprime_not_in_cone(arrays)
      columns_not_truth = svj.bdt_feature_columns(not_truth)

      

      for k in range(len(variables)):
         #signal[variables[k]] = columns.arrays[variables[k]]
         signal_truth[variables[k]] = columns_truth.arrays[variables[k]]
         signal_not_truth[variables[k]] = columns_not_truth.arrays[variables[k]]
         signal[variables[k]] = columns.arrays[variables[k]]
      #np.savez('signal_files/mz_'+mz[i]+'_rinv_'+rinv[j]+'.npz', **signal)
      np.savez('signal_files/truth/mz_'+mz[i]+'_rinv_'+rinv[j]+'.npz', **signal)
      np.savez('signal_files/truth/mz_'+mz[i]+'_rinv_'+rinv[j]+'_truth.npz', **signal_truth)
      np.savez('signal_files/truth/mz_'+mz[i]+'_rinv_'+rinv[j]+'_not_truth.npz', **signal_not_truth)'''


'''for i in range(len(mz)):
   print(umd_files+str(mz[i])+'_mdark5_rinv0.3.root')
   print('mz_'+mz[i]+'_rinv_0p3')
   arrays = svj.open_root(umd_files+str(mz[i])+'_mdark5_rinv0.3.root')
   arrays = svj.cr_filter_preselection(arrays)
   columns = svj.bdt_feature_columns(arrays)

   for k in range(len(variables)):
      signal[variables[k]] = columns.arrays[variables[k]]
   np.savez('signal_files/mz_'+mz[i]+'_mdark5_rinv_0p3.npz', **signal)'''
