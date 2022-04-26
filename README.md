# NSP-Grant-Proposal

 Measuring depth of anaesthesia using entropy of EEG signals, and body vitals. 
We use the following parameters from pateints undergoing surgery with the anaesthesia: 
<ul>
  <li>Raw EEG < \li>
  <li>Heart Rate < \li>
  <li>0_2 Saturation < \li>
  <li>Respiration rate < \li> 
< \ul>
We use the data of 4 pateints from the study (https://journals.lww.com/anesthesia-analgesia/Fulltext/2012/03000/University_of_Queensland_Vital_Signs_Dataset_.15.aspx). We finnaly give one score \epsilon [0,1] which indicates how awake the patient is. We Compare our value to a standard parameter BIS.

## Our approach:
_These are the proceedure we have used_
<ol>
  <li>Filtering and Preprocessing
     <ul>
      <li>Removal of NaN values and outlier points( ```|x-\mu|\geq 4\signma```)  </li>
      <li>Removal of 50Hz Line noise using spectral interpolation  </li>
     </ul>
   <\li>
  <li>Computation of Entropies
   <ul>
      <li>Hilbert Huang Entropy( using EMD and Hilbert transform)  </li>
      <li>Approximate Entropy  </li>
     </ul>
    _Both of these are computed at two frequency bands.State entropy (SE) is the entropy at the frequency band ranging from 0.8 to 32 Hz; Response entropy (RE) is calculated across the frequency band 0.8 to 47 Hz_
    </li>  
   <li>Useful Spectral parameters
    <ul>
      <li>Total Power in \alpha band  </li>
      <li>Total Power in \delta band  </li>
      <li>Delta Ratio (Ratio of power in 8Hz-30Hz band and 0Hz-4Hz range)<\li>
      <li>SPF-90 ( The frequency f such that 90% of the power lies in the band 0Hz-fHz)
     </ul>
    </li>
  <li>Neural Network
     To find the right combination of these parameters, We train a basic Neurla Network with 1 hidden layer. The Neural Network is (11 X 5 X 1), with L1 regularisation in the First Layer(so that parameters that are partciularly not useful are discarded) 
  </li>
</ol>

