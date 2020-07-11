$openvino_download_link = "http://registrationcenter-download.intel.com/akdlm/irc_nas/16359/w_openvino_toolkit_p_2020.1.033.exe"
$target_file = "$HOME\Downloads\w_openvino_toolkit_p_2020.1.033.exe"
(new-object System.Net.WebClient).DownloadFile("$openvino_download_link", "$target_file")
& "$target_file"