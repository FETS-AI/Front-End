#include <map>

#include "cbicaCmdParser.h"
#include "cbicaLogging.h"
#include "cbicaITKSafeImageIO.h"
#include "cbicaUtilities.h"
#include "cbicaITKUtilities.h"
#include "CaPTkGUIUtils.h"

int runCollaboratorTraining(const std::string &fullCommandToRunWithArgs)
{
  auto returnCode = std::system(fullCommandToRunWithArgs.c_str());
  if (returnCode != 0)
  {
    if (returnCode == 154)
    {
      std::cout << "Special case, where the collaborator failing is expected, so automatically restarting.\n";
      return runCollaboratorTraining(fullCommandToRunWithArgs);
    }
  }
}

int main(int argc, char** argv)
{
  cbica::CmdParser parser(argc, argv, "FeTS_CLI");

  auto hardcodedNativeModelWeightPath = getCaPTkDataDir() + "/fets";
  auto allArchs = cbica::subdirectoriesInDirectory(hardcodedNativeModelWeightPath);
  std::string allArchsString;
  for (size_t i = 0; i < allArchs.size(); i++)
  {
    allArchsString += allArchs[i] + ",";
  }
  allArchsString.pop_back();

  std::string dataDir, modelName, loggingDir, colName, archs = "3dresunet", fusionMethod = "STAPLE", hardcodedPlanName = "fets_phase2_2";

  parser.addRequiredParameter("d", "dataDir", cbica::Parameter::DIRECTORY, "Dir with Read/Write access", "Input data directory");
  parser.addRequiredParameter("t", "training", cbica::Parameter::BOOLEAN, "0 or 1", "Whether performing training or inference", "1==Train and 0==Inference");
  parser.addOptionalParameter("tp", "trainPlan", cbica::Parameter::BOOLEAN, "YAML file", "Training plan", "Defaults to '" + hardcodedPlanName + "'");
  parser.addOptionalParameter("L", "LoggingDir", cbica::Parameter::DIRECTORY, "Dir with write access", "Location of logging directory");
  parser.addOptionalParameter("a", "archs", cbica::Parameter::STRING, allArchsString, "The architecture(s) to infer/train on", "Only a single architecture is supported for training", "Comma-separated values for multiple options", "Defaults to: " + archs);
  parser.addOptionalParameter("lF", "labelFuse", cbica::Parameter::STRING, "STAPLE,ITKVoting,SIMPLE,MajorityVoting", "The label fusion strategy to follow for multi-arch inference", "Comma-separated values for multiple options", "Defaults to: " + fusionMethod);
  parser.addOptionalParameter("g", "gpu", cbica::Parameter::BOOLEAN, "0-1", "Whether to run the process on GPU or not", "Defaults to '0'");
  parser.addOptionalParameter("c", "colName", cbica::Parameter::STRING, "", "Common name of collaborator", "Required for training");
  // parser.addOptionalParameter("vp", "valPatch", cbica::Parameter::BOOLEAN, "0-1", "Whether to perform per-patch validation or not", "Used for training, defaults to '0'");

  parser.addApplicationDescription("This is the CLI interface for FeTS");
  parser.addExampleUsage("-d /path/DataForFeTS -a deepMedic,nnUNet -lF STAPLE,ITKVoting,SIMPLE -g 1 -t 0", "This command performs inference using deepMedic,nnUNet using multiple fusion strategies on GPU and saves in data directory");
  parser.addExampleUsage("-d /path/DataForFeTS -t 1 -g 1 -c upenn", "This command starts training performs inference using deepMedic,nnUNet using multiple fusion strategies on GPU and saves in data directory");
  
  bool gpuRequested = false, trainingRequested = false, patchValidation = true;

  parser.getParameterValue("d", dataDir);
  parser.getParameterValue("t", trainingRequested);

  if (parser.isPresent("L"))
  {
    parser.getParameterValue("L", loggingDir);
  }
  else
  {
    loggingDir = dataDir + "/logs";
    std::cout << "Using the following directory as logging directory: " << loggingDir << "\n";
    cbica::createDir(loggingDir);
  }

  if (trainingRequested)
  {
    if (parser.isPresent("c"))
    {
      parser.getParameterValue("c", colName);
    }
    else
    {
      std::cerr << "Collaborator name is required to beging training; please specify this using '-c'.\n";
      return EXIT_FAILURE;
    }
    // if (parser.isPresent("vp"))
    // {
    //   parser.getParameterValue("vp", patchValidation);
    // }
  }
  else
  {
    if (parser.isPresent("a"))
    {
      parser.getParameterValue("a", archs);
    }
    else
    {
      std::cerr << "Please specify at least 2 architectures on which to perform inference.\n";
    }
    if (parser.isPresent("lF"))
    {
      parser.getParameterValue("lF", fusionMethod);
    }
  }
  if (parser.isPresent("g"))
  {
    parser.getParameterValue("g", gpuRequested);
  }

  std::string device_arg = " -md ";
  if (gpuRequested)
  {
    device_arg += "cuda";
  }
  else
  {
    device_arg += "cpu";
  }


  // convert everything to lower-case for easier comparison
  std::transform(archs.begin(), archs.end(), archs.begin(), ::tolower);
  std::transform(fusionMethod.begin(), fusionMethod.end(), fusionMethod.begin(), ::tolower);

  auto fetsApplicationPath = cbica::getExecutablePath();
  auto deepMedicExe = getApplicationPath("DeepMedic");

  auto archs_split = cbica::stringSplit(archs, ",");
  auto fusion_split = cbica::stringSplit(fusionMethod, ",");

  auto subjectDirs = cbica::subdirectoriesInDirectory(dataDir);

  if (trainingRequested && (archs_split.size() > 1))
  {
    std::cerr << "Training cannot be currently be performed on more than 1 architecture.\n";
    return EXIT_FAILURE;
  }

  std::string 
    hardcodedOpenFLPath = fetsApplicationPath + "/OpenFederatedLearning/",
    hardcodedLabelFusionPath = fetsApplicationPath + "/LabelFusion/fusion_run",
    hardcodedModelWeightPath = hardcodedOpenFLPath + "/bin/federations/weights/", // start with the common location
    //hardcodedNativeModelWeightPath = hardcodedOpenFLPath + "/bin/federations/weights/native/", // the native weights are going in fets_data_dir/fets
    hardcodedPythonPath = hardcodedOpenFLPath + "/venv/bin/python", // this needs to change for Windows (wonder what happens for macOS?)
    hardcodedPythonPath_fusion = fetsApplicationPath + "/LabelFusion/venv/bin/python"; // this needs to change for Windows (wonder what happens for macOS?)
#if WIN32
  hardcodedPythonPath = hardcodedOpenFLPath + "/venv/python.exe";
#endif

  auto pythonEnvironmentFound = false;
  if (cbica::isFile(hardcodedPythonPath))
  {
    pythonEnvironmentFound = true;
  }

  if (!trainingRequested)
  {
    std::string subjectsWithMissingModalities, subjectsWithErrors; // string to store error cases
        
    std::cout << "Starting subject directory iteration...\n";
    for (size_t s = 0; s < subjectDirs.size(); s++) // iterate through all subjects
    {
      if (subjectDirs[s].find("logs") == std::string::npos)
      {
        auto currentSubjectIsProblematic = false;
        std::string file_t1gd, file_t1, file_t2, file_flair;

        auto filesInDir = cbica::filesInDirectory(dataDir + "/" + subjectDirs[s]); // get files in current subject directory
        // iterate through all files and pick up individual modalities
        for (size_t f = 0; f < filesInDir.size(); f++)
        {
          if (file_t1gd.empty())
          {
            if ((filesInDir[f].find("_t1ce.nii.gz") != std::string::npos) || (filesInDir[f].find("_t1gd.nii.gz") != std::string::npos))
            {
              file_t1gd = filesInDir[f];
            }
          }
          if (file_t1.empty())
          {
            if (filesInDir[f].find("_t1.nii.gz") != std::string::npos)
            {
              file_t1 = filesInDir[f];
            }
          }
          if (file_t2.empty())
          {
            if (filesInDir[f].find("_t2.nii.gz") != std::string::npos)
            {
              file_t2 = filesInDir[f];
            }
          }
          if (file_flair.empty())
          {
            if ((filesInDir[f].find("_flair.nii.gz") != std::string::npos) || (filesInDir[f].find("_fl.nii.gz") != std::string::npos))
            {
              file_flair = filesInDir[f];
            }
          }
        }

        // ensure problematic cases are detected
        if (file_t1gd.empty())
        {
          subjectsWithMissingModalities += subjectDirs[s] + ",t1ce\n";
          currentSubjectIsProblematic = true;
        }
        if (file_t1.empty())
        {
          subjectsWithMissingModalities += subjectDirs[s] + ",t1\n";
          currentSubjectIsProblematic = true;
        }
        if (file_t2.empty())
        {
          subjectsWithMissingModalities += subjectDirs[s] + ",t2\n";
          currentSubjectIsProblematic = true;
        }
        if (file_flair.empty())
        {
          subjectsWithMissingModalities += subjectDirs[s] + ",flair\n";
          currentSubjectIsProblematic = true;
        }

        if (!currentSubjectIsProblematic) // proceed only if all modalities for the current subject are present
        {
          std::cout << "= Starting inference for subject: " << subjectDirs[s] << "\n";
          for (size_t a = 0; a < archs_split.size(); a++) // iterate through all requested architectures
          {
            if (archs_split[a] == "deepmedic") // special case 
            {
              std::cout << "== Starting inference using DeepMedic...\n";
              auto brainMaskFile = dataDir + "/" + subjectDirs[s] + "/" + subjectDirs[s] + "_deepmedic_seg.nii.gz";
              auto fileToCheck_2 = dataDir + "/" + subjectDirs[s] + "/SegmentationsForQC/" + subjectDirs[s] + "_deepmedic_seg.nii.gz";
              if (!(cbica::isFile(brainMaskFile) || cbica::isFile(fileToCheck_2)))
              {
                auto dm_tempOut = dataDir + "/" + subjectDirs[s] + "/dmOut/mask.nii.gz";
                auto fullCommand = deepMedicExe + " -md " + hardcodedNativeModelWeightPath + "/deepMedic/saved_models/brainTumorSegmentation/ " +
                  "-i " + file_t1 + "," +
                  file_t1gd + "," +
                  file_t2 + "," +
                  file_flair + " -o " +
                  dm_tempOut;

                if (std::system(fullCommand.c_str()) != 0)
                {
                  std::cerr << "=== Couldn't complete the inference for deepmedic for subject " << subjectDirs[s] << ".\n";
                  subjectsWithErrors += subjectDirs[s] + ",inference,deepmedic\n";
                }
                else
                {
                  cbica::copyFile(dm_tempOut, brainMaskFile);
                }
              }
            } // deepmedic check
            else
            {
              auto fullCommandToRun = hardcodedPythonPath + " " + hardcodedOpenFLPath + "/bin/run_inference_from_flplan.py";
              auto args = " -d " + dataDir + device_arg + " -ld " + loggingDir + " -ip " + subjectDirs[s];
              if (pythonEnvironmentFound)
              {
                // check for all other models written in pytorch here

                // check between different architectures
                if (archs_split[a] == "3dunet")
                {
                  // this is currently not defined
                }
                else if (archs_split[a] == "3dresunet")
                {
                  std::cout << "3DResUNet inference is disabled for this release.\n";
                  //auto fileNameToCheck = subjectDirs[s] + "_resunet_seg.nii.gz";
                  //auto fileToCheck_1 = dataDir + "/" + subjectDirs[s] + "/" + fileNameToCheck;
                  //auto fileToCheck_2 = dataDir + "/" + subjectDirs[s] + "/SegmentationsForQC/" + fileNameToCheck;
                  //if (!(cbica::isFile(fileToCheck_1) || cbica::isFile(fileToCheck_2))) // don't run if file is present
                  //{
                  //  std::cout << "== Starting inference using 3DResUNet...\n";
                  //  hardcodedPlanName = "pt_3dresunet_brainmagebrats";
                  //  auto hardcodedModelName = hardcodedPlanName + "_best.pbuf";
                  //  if (!cbica::isFile((hardcodedModelWeightPath + "/" + hardcodedModelName))) // in case the "best" model is not present, use the "init" model that is distributed with FeTS installation
                  //  {
                  //    hardcodedModelName = hardcodedPlanName + "_init.pbuf";
                  //    if (!cbica::isFile((hardcodedModelWeightPath + "/" + hardcodedModelName)))
                  //    {
                  //      std::cerr << "=== A compatible model weight file for the architecture '" << archs_split[a] << "' was not found. Please contact admin@fets.ai for help.\n";
                  //    }
                  //  }

                  //  auto args_to_run = args + " -mwf " + hardcodedModelName
                  //    + " -p " + hardcodedPlanName + ".yaml";
                  //  //<< "-mwf" << hardcodedModelWeightPath // todo: doing customized solution above - change after model weights are using full paths for all

                  //  if (std::system((fullCommandToRun + " " + args_to_run).c_str()) != 0)
                  //  {
                  //    std::cerr << "=== Couldn't complete the inference for 3dresunet for subject " << subjectDirs[s] << ".\n";
                  //    subjectsWithErrors += subjectDirs[s] + ",inference,3dresunet\n";
                  //  }
                  //} // end of previous run file check
                } // end of 3dresunet check
                else
                {
                  std::string hardcodedPlanName;
                  if (archs_split[a].find("nnunet") != std::string::npos)
                  {
                    hardcodedPlanName = "nnunet";
                    std::cout << "== Starting inference using nnUNet...\n";
                  }
                  else if (archs_split[a].find("deepscan") != std::string::npos)
                  {
                    hardcodedPlanName = "deepscan";
                    std::cout << "== Starting inference using DeepScan...\n";
                  }
                  if (!hardcodedPlanName.empty())
                  {
                    auto fileNameToCheck = subjectDirs[s] + "_" + hardcodedPlanName + "_seg.nii.gz";
                    auto fileToCheck_1 = dataDir + "/" + subjectDirs[s] + "/" + fileNameToCheck;
                    auto fileToCheck_2 = dataDir + "/" + subjectDirs[s] + "/SegmentationsForQC/" + fileNameToCheck;
                    if (!(cbica::isFile(fileToCheck_1) || cbica::isFile(fileToCheck_2))) // don't run if file is present
                    {
                      // structure according to what is needed - might need to create a function that can call run_inference_from_flplan for different hardcodedModelName
                      auto args_to_run = args + " -nmwf " + hardcodedNativeModelWeightPath + "/" + hardcodedPlanName // <abs path to folder containing all model weights folders> 
                        + " -p " + hardcodedPlanName + "_inference.yaml"
                        + " -pwai";

                      /// remove before final packaging
                      std::cerr << "=== \n=== Command to run: \n" << fullCommandToRun + " " + args_to_run << "\n===\n";

                      if (std::system((fullCommandToRun + " " + args_to_run).c_str()) != 0)
                      {
                        std::cerr << "=== Couldn't complete the inference for " << archs_split[a] << " for subject " << subjectDirs[s] << ".\n";
                        subjectsWithErrors += subjectDirs[s] + ",inference," + archs_split[a] + "\n";
                      }
                    } // end of previous run file check
                  } // end of hardcodedPlanName check
                } // end of non-3dresunet check
              } // end of python check
            } // end of non-DM archs check
          } // end of archs_split

          /// fusion 
          if (pythonEnvironmentFound)
          {
            if (cbica::isFile(hardcodedLabelFusionPath))
            {
              std::cout << "== Starting label fusion...\n";
              auto filesInSubjectDir = cbica::filesInDirectory(dataDir + "/" + subjectDirs[s]);
              auto labelFusion_command = hardcodedPythonPath_fusion + " " + hardcodedLabelFusionPath + " ";
              std::string filesForFusion, dataForSegmentation = dataDir + "/" + subjectDirs[s] + "/SegmentationsForQC/";
              cbica::createDir(dataForSegmentation);
              auto dm_folder = dataDir + "/" + subjectDirs[s] + "/dmOut";
              if (cbica::isDir(dm_folder))
              {
                cbica::copyDir(dm_folder, dataForSegmentation);
                cbica::removeDirectoryRecursively(dm_folder, true);
              }

              for (size_t f = 0; f < filesInSubjectDir.size(); f++)
              {
                if (filesInSubjectDir[f].find("_seg.nii.gz") != std::string::npos) // find all files that have "_seg.nii.gz" in file name
                {
                  if (filesInSubjectDir[f].find("final") == std::string::npos) // only do fusion for the files where "final" is not present
                  {
                    auto fileToCopy = dataForSegmentation + cbica::getFilenameBase(filesInSubjectDir[f]) + ".nii.gz";
                    cbica::copyFile(filesInSubjectDir[f], fileToCopy);
                    filesForFusion += fileToCopy + ",";
                    std::remove(filesInSubjectDir[f].c_str());
                  }
                }
              } // files loop in subject directory
              filesInSubjectDir = cbica::filesInDirectory(dataForSegmentation);
              for (size_t f = 0; f < filesInSubjectDir.size(); f++)
              {
                auto fileToCopy = dataForSegmentation + cbica::getFilenameBase(filesInSubjectDir[f]) + ".nii.gz";
                if (filesInSubjectDir[f].find("fused") == std::string::npos) // only consider those files for fusion that are arch outputs
                {
                  filesForFusion += fileToCopy + ",";
                }
              } // files loop in subject directory

              if (!filesForFusion.empty())
              {
                filesForFusion.pop_back(); // remove last ","
              }

              for (size_t f = 0; f < fusion_split.size(); f++)
              {
                auto final_fused_file = dataForSegmentation + "/" + subjectDirs[s] + "_fused_" + fusion_split[f] + "_seg.nii.gz";
                auto full_fusion_command = labelFusion_command + "-inputs " + filesForFusion + " -classes 0,1,2,4 " // this needs to change after different segmentation algorithms are put in place
                  + " -method " + fusion_split[f] + " -output " + final_fused_file;
                if (std::system(full_fusion_command.c_str()) != 0)
                {
                  std::cerr << "=== Something went wrong with fusion for subject '" << subjectDirs[s] << "' using fusion method '" << fusion_split[f] << "'\n";
                  subjectsWithErrors += subjectDirs[s] + ",fusion," + fusion_split[f] + "\n";
                }
              }
            } // end of label fusion script check
          } // end of python check
        } // end of currentSubjectIsProblematic 
      } // end of logs check
    } // end of subjectDirs

    // provide error message
    if (!subjectsWithMissingModalities.empty())
    {
      std::cerr << "\nThe following subjects did not have all the 4 structural modalities to proceed with preprocessing:\nSubjectID,Modality\n" << subjectsWithMissingModalities;
    }
    if (!subjectsWithErrors.empty())
    {
      std::cerr << "\nThe following subjects were problematic:\nSubjectID,Application,Algorithm\n" << subjectsWithErrors;
    }
  } // end of trainingRequested check
  else // for training
  {
    /// start validation of nnunet/deepscan/deepmedic on all validation cases
    auto split_info_val = dataDir + "/split_info/fets_phase2_split_1/val.csv", // revisit in case we change split in the future
      validation_to_send = dataDir + "/validation.yaml",
      validation_internal = dataDir + "/validation_internal.yaml";

    if (!cbica::isFile(hardcodedPythonPath))
    {
      std::cerr << "The python virtual environment was not found, please refer to documentation to initialize it.\n";
      return EXIT_FAILURE;
    }

    if (!cbica::fileExists(split_info_val))
    {
      auto full_plan_path = hardcodedOpenFLPath + hardcodedPlanName;
      auto command_to_run = hardcodedPythonPath + " " + hardcodedOpenFLPath + "submodules/fets_ai/Algorithms/fets/bin/initialize_split_info.py -pp " + full_plan_path + " -dp " + dataDir;
      if (std::system(command_to_run.c_str()) != 0)
      {
        std::cerr << "Initialize split did not work, continuing with validation.\n";
      }
    }
    
    if (cbica::fileExists(split_info_val))
    {
      std::ifstream file(split_info_val.c_str());
      bool firstRow = true;
      int row_index = -1;
      auto regions_of_interest = { "WT", "TC", "ET" },
        measures_of_interest = { "Dice", "Hausdorff95", "Sensitivity", "Specificity" };

      auto yaml_config_to_send = YAML::Node();
      auto yaml_config_internal = YAML::Node();

      if (cbica::isFile(validation_internal)) // load previous internal validation file
      {
        yaml_config_internal = YAML::LoadFile(validation_internal);
      }

      while (file)
      {
        std::string line;
        std::getline(file, line);
        // fix line ending problems 
        std::remove_copy(line.begin(), line.end(), line.begin(), '\r');
        std::stringstream lineStream(line);
        std::vector<std::string> row;
        std::string cell;
        while (getline(lineStream, cell, ','))
        {
          if (row_index > -1)
          {
            auto subject_id = cell;
            auto subject_index_str = std::to_string(row_index);

            bool previous_validation_file_is_okay = true;

            if (yaml_config_internal[subject_id]) // check if subject is present in internal validation file
            {
              yaml_config_to_send[subject_index_str] = yaml_config_internal[subject_id]; // if present, take all stats from there
              auto to_check = yaml_config_internal[subject_id]["WT"];
              if (!yaml_config_internal[subject_id]["WT"]["Sensitivity"]) // check if sensitivity is present for subject
              {
                previous_validation_file_is_okay = false;
              }
            }
            else
            {
              previous_validation_file_is_okay = false;
            }

            if (!previous_validation_file_is_okay)
            {
              auto current_subject_folder = dataDir + "/" + subject_id;
              auto final_seg = current_subject_folder + "/" + subject_id + "_final_seg.nii.gz";
              std::map< std::string, std::string > archs_to_check;
              archs_to_check["deepmedic"] = current_subject_folder + "/SegmentationsForQC/" + subject_id + "_deepmedic_seg.nii.gz";
              archs_to_check["nnunet"] = current_subject_folder + "/SegmentationsForQC/" + subject_id + "_nnunet_seg.nii.gz";
              archs_to_check["deepscan"] = current_subject_folder + "/SegmentationsForQC/" + subject_id + "_deepscan_seg.nii.gz";
              if (!cbica::isFile(final_seg))
              {
                std::cerr << "The subject '" << subject_id << "' does not have a final_seg file present.\n";
              }
              else
              {
                using DefaultImageType = itk::Image< unsigned int, 3 >;
                auto final_seg_image = cbica::ReadImage< DefaultImageType >(final_seg);
                for (auto& current_arch : archs_to_check)
                {
                  if (cbica::isFile(current_arch.second))
                  {
                    auto image_to_check = cbica::ReadImage< DefaultImageType >(current_arch.second);

                    auto stats = cbica::GetBraTSLabelStatistics< DefaultImageType >(final_seg_image, image_to_check);

                    for (auto& region : regions_of_interest)
                    {
                      for (auto& measure : measures_of_interest)
                      {
                        yaml_config_to_send[subject_index_str][current_arch.first][region][measure] = stats[region][measure];
                        yaml_config_internal[subject_id][current_arch.first][region][measure] = stats[region][measure];
                      } // end measure loop
                    } // end region loop
                  } // end file-check loop
                } // end arch-loop
              } // end final_seg check 
            } // end internal validation check loop
          } // end header check if-loop
          row_index++; // increment subject id counter
        } // end csv-read while loop
      }
      std::ofstream fout_int(validation_internal);
      fout_int << yaml_config_internal; // dump it back into the file
      fout_int.close();

      std::ofstream fout(validation_to_send);
      fout << yaml_config_to_send; // dump it back into the file
      fout.close();
    }

    std::string args = " -d " + dataDir + " -ld " + loggingDir + " -col " + colName + device_arg,
      hardcodedModelName;

    if (!patchValidation)
    {
      args += " -vwop";
    }
    
    {
      std::cout << "Starting model validation of 3DResUNet trained on BraTS20 training data...\n";

      // brats20 model validation      
      std::string fullCommandToRun = hardcodedPythonPath + " " + fetsApplicationPath;
      fullCommandToRun += "/OpenFederatedLearning/bin/run_fets_validation.py";

      auto temp_args = args + " -p fets_phase1_validate_full_brats_trained_model_1.yaml";

      if (std::system((fullCommandToRun + " " + temp_args).c_str()) != 0)
      {
        std::cerr << "Couldn't complete the BraTS20 model validation task, please email admin@fets.ai\n";
      }
    }

    std::string fullCommandToRun = hardcodedPythonPath + " " + fetsApplicationPath;
    fullCommandToRun += "/OpenFederatedLearning/bin/run_collaborator_from_flplan.py";

    auto temp_args = args + " -p " + hardcodedPlanName + ".yaml" + " -bsuf " + validation_to_send + " -nlo";

    std::cout << "Starting training...\n";

    if (runCollaboratorTraining(fullCommandToRun + " " + temp_args) != 0)
    {
      std::cerr << "Couldn't complete the training task, please email admin@fets.ai\n";
      return EXIT_FAILURE;
    }

  } // end of trainingRequested check

  std::cout << "Finished.\n";

  return EXIT_SUCCESS;
}


