#!/usr/bin/env bash

# You can change the default directory where the temporary file will be stored during parsing.
export JSON_FILE_DIRECTORY=/tmp

# You can change the protocol, and the endpoint, according to your configuration.
export PROMETHEUS_PROTOCOL="http"
export PROMETHEUS_ENDPOINT="metrics"

if [[ ! -z "${2}" ]]; then
  PROMETHEUS_HOST="${2}"
else
  PROMETHEUS_HOST="localhost"
fi

if [[ ! -z "${3}" ]]; then
  PROMETHEUS_PORT="${3}"
else
  PROMETHEUS_PORT="8000"
fi

cd ${JSON_FILE_DIRECTORY} || exit 1

# If file is older than four (4) minutes, refresh the file.
if [[ ! -f fsp-observer-output-clean_${PROMETHEUS_HOST}_${PROMETHEUS_PORT}.json || $( find fsp-observer-output-clean_${PROMETHEUS_HOST}_${PROMETHEUS_PORT}.json -mmin +4 ) ]]; then
  if [[ -f fsp-observer-output-clean_${PROMETHEUS_HOST}_${PROMETHEUS_PORT}.json ]]; then
    rm -f fsp-observer-output-clean_${PROMETHEUS_HOST}_${PROMETHEUS_PORT}.json
  fi
  
  if [[ -f zbx_fsp_observer_json_query_${PROMETHEUS_HOST}_${PROMETHEUS_PORT}.lock ]]; then
    while true
    do
      sleep 5
      
      if [[ ! -f zbx_fsp_observer_json_query_${PROMETHEUS_HOST}_${PROMETHEUS_PORT}.lock && -f fsp-observer-output-clean_${PROMETHEUS_HOST}_${PROMETHEUS_PORT}.json && ! $( find fsp-observer-output-clean_${PROMETHEUS_HOST}_${PROMETHEUS_PORT}.json -mmin +4 ) ]]; then
        break
      fi
    done
    
    output=`cat fsp-observer-output-clean_${PROMETHEUS_HOST}_${PROMETHEUS_PORT}.json`

    echo ${output}
    
    exit 0
  fi
  
  echo $$ > zbx_fsp_observer_json_query_${PROMETHEUS_HOST}_${PROMETHEUS_PORT}.lock
  
  export PROMETHEUS_HOST_FULL_URI="${PROMETHEUS_PROTOCOL}://${PROMETHEUS_HOST}:${PROMETHEUS_PORT}/${PROMETHEUS_ENDPOINT}"

  curl -s -o fsp-observer-output_${PROMETHEUS_HOST}_${PROMETHEUS_PORT}.txt "${PROMETHEUS_HOST_FULL_URI}"

  sed -i '/^[[:blank:]]*#/d' fsp-observer-output_${PROMETHEUS_HOST}_${PROMETHEUS_PORT}.txt
  
  # flare_fsp_submit_ok_total
  echo -e "flare_fsp_submit_ok_total_submit1:$( grep "flare_fsp_submit_ok_total.*submit1" fsp-observer-output_${PROMETHEUS_HOST}_${PROMETHEUS_PORT}.txt | awk '{print $NF}' )" > fsp-observer-output-clean_${PROMETHEUS_HOST}_${PROMETHEUS_PORT}.txt
  echo -e "flare_fsp_submit_ok_total_submit2_ftso:$( grep "flare_fsp_submit_ok_total.*submit2.*ftso" fsp-observer-output_${PROMETHEUS_HOST}_${PROMETHEUS_PORT}.txt | awk '{print $NF}' )" >> fsp-observer-output-clean_${PROMETHEUS_HOST}_${PROMETHEUS_PORT}.txt
  echo -e "flare_fsp_submit_ok_total_submit2_fdc:$( grep "flare_fsp_submit_ok_total.*submit2.*fdc" fsp-observer-output_${PROMETHEUS_HOST}_${PROMETHEUS_PORT}.txt | awk '{print $NF}' )" >> fsp-observer-output-clean_${PROMETHEUS_HOST}_${PROMETHEUS_PORT}.txt
  echo -e "flare_fsp_submit_ok_total_signatures_ftso:$( grep "flare_fsp_submit_ok_total.*signatures.*ftso" fsp-observer-output_${PROMETHEUS_HOST}_${PROMETHEUS_PORT}.txt | awk '{print $NF}' )" >> fsp-observer-output-clean_${PROMETHEUS_HOST}_${PROMETHEUS_PORT}.txt
  echo -e "flare_fsp_submit_ok_total_signatures_fdc:$( grep "flare_fsp_submit_ok_total.*signatures.*fdc" fsp-observer-output_${PROMETHEUS_HOST}_${PROMETHEUS_PORT}.txt | awk '{print $NF}' )" >> fsp-observer-output-clean_${PROMETHEUS_HOST}_${PROMETHEUS_PORT}.txt
  # flare_fsp_submit_late_total
  echo -e "flare_fsp_submit_late_total_submit1:$( grep "flare_fsp_submit_late_total.*submit1" fsp-observer-output_${PROMETHEUS_HOST}_${PROMETHEUS_PORT}.txt | awk '{print $NF}' )" >> fsp-observer-output-clean_${PROMETHEUS_HOST}_${PROMETHEUS_PORT}.txt
  echo -e "flare_fsp_submit_late_total_submit2_ftso:$( grep "flare_fsp_submit_late_total.*submit2.*ftso" fsp-observer-output_${PROMETHEUS_HOST}_${PROMETHEUS_PORT}.txt | awk '{print $NF}' )" >> fsp-observer-output-clean_${PROMETHEUS_HOST}_${PROMETHEUS_PORT}.txt
  echo -e "flare_fsp_submit_late_total_submit2_fdc:$( grep "flare_fsp_submit_late_total.*submit2.*fdc" fsp-observer-output_${PROMETHEUS_HOST}_${PROMETHEUS_PORT}.txt | awk '{print $NF}' )" >> fsp-observer-output-clean_${PROMETHEUS_HOST}_${PROMETHEUS_PORT}.txt
  echo -e "flare_fsp_submit_late_total_signatures_ftso:$( grep "flare_fsp_submit_late_total.*signatures.*ftso" fsp-observer-output_${PROMETHEUS_HOST}_${PROMETHEUS_PORT}.txt | awk '{print $NF}' )" >> fsp-observer-output-clean_${PROMETHEUS_HOST}_${PROMETHEUS_PORT}.txt
  echo -e "flare_fsp_submit_late_total_signatures_fdc:$( grep "flare_fsp_submit_late_total.*signatures.*fdc" fsp-observer-output_${PROMETHEUS_HOST}_${PROMETHEUS_PORT}.txt | awk '{print $NF}' )" >> fsp-observer-output-clean_${PROMETHEUS_HOST}_${PROMETHEUS_PORT}.txt
  # flare_fsp_submit_early_total
  echo -e "flare_fsp_submit_early_total_submit1:$( grep "flare_fsp_submit_early_total.*submit1" fsp-observer-output_${PROMETHEUS_HOST}_${PROMETHEUS_PORT}.txt | awk '{print $NF}' )" >> fsp-observer-output-clean_${PROMETHEUS_HOST}_${PROMETHEUS_PORT}.txt
  echo -e "flare_fsp_submit_early_total_submit2_ftso:$( grep "flare_fsp_submit_early_total.*submit2.*ftso" fsp-observer-output_${PROMETHEUS_HOST}_${PROMETHEUS_PORT}.txt | awk '{print $NF}' )" >> fsp-observer-output-clean_${PROMETHEUS_HOST}_${PROMETHEUS_PORT}.txt
  echo -e "flare_fsp_submit_early_total_submit2_fdc:$( grep "flare_fsp_submit_early_total.*submit2.*fdc" fsp-observer-output_${PROMETHEUS_HOST}_${PROMETHEUS_PORT}.txt | awk '{print $NF}' )" >> fsp-observer-output-clean_${PROMETHEUS_HOST}_${PROMETHEUS_PORT}.txt
  echo -e "flare_fsp_submit_early_total_signatures_ftso:$( grep "flare_fsp_submit_early_total.*signatures.*ftso" fsp-observer-output_${PROMETHEUS_HOST}_${PROMETHEUS_PORT}.txt | awk '{print $NF}' )" >> fsp-observer-output-clean_${PROMETHEUS_HOST}_${PROMETHEUS_PORT}.txt
  echo -e "flare_fsp_submit_early_total_signatures_fdc:$( grep "flare_fsp_submit_early_total.*signatures.*fdc" fsp-observer-output_${PROMETHEUS_HOST}_${PROMETHEUS_PORT}.txt | awk '{print $NF}' )" >> fsp-observer-output-clean_${PROMETHEUS_HOST}_${PROMETHEUS_PORT}.txt
  # flare_fsp_submit_missing_total
  echo -e "flare_fsp_submit_missing_total_submit1:$( grep "flare_fsp_submit_missing_total.*submit1" fsp-observer-output_${PROMETHEUS_HOST}_${PROMETHEUS_PORT}.txt | awk '{print $NF}' )" >> fsp-observer-output-clean_${PROMETHEUS_HOST}_${PROMETHEUS_PORT}.txt
  echo -e "flare_fsp_submit_missing_total_submit2_ftso:$( grep "flare_fsp_submit_missing_total.*submit2.*ftso" fsp-observer-output_${PROMETHEUS_HOST}_${PROMETHEUS_PORT}.txt | awk '{print $NF}' )" >> fsp-observer-output-clean_${PROMETHEUS_HOST}_${PROMETHEUS_PORT}.txt
  echo -e "flare_fsp_submit_missing_total_submit2_fdc:$( grep "flare_fsp_submit_missing_total.*submit2.*fdc" fsp-observer-output_${PROMETHEUS_HOST}_${PROMETHEUS_PORT}.txt | awk '{print $NF}' )" >> fsp-observer-output-clean_${PROMETHEUS_HOST}_${PROMETHEUS_PORT}.txt
  echo -e "flare_fsp_submit_missing_total_signatures_ftso:$( grep "flare_fsp_submit_missing_total.*signatures.*ftso" fsp-observer-output_${PROMETHEUS_HOST}_${PROMETHEUS_PORT}.txt | awk '{print $NF}' )" >> fsp-observer-output-clean_${PROMETHEUS_HOST}_${PROMETHEUS_PORT}.txt
  echo -e "flare_fsp_submit_missing_total_signatures_fdc:$( grep "flare_fsp_submit_missing_total.*signatures.*fdc" fsp-observer-output_${PROMETHEUS_HOST}_${PROMETHEUS_PORT}.txt | awk '{print $NF}' )" >> fsp-observer-output-clean_${PROMETHEUS_HOST}_${PROMETHEUS_PORT}.txt
  # flare_fsp_address_balance_wei
  echo -e "flare_fsp_address_balance_wei_submit:$( grep "flare_fsp_address_balance_wei.*submit\"" fsp-observer-output_${PROMETHEUS_HOST}_${PROMETHEUS_PORT}.txt | awk '{print $NF}' )" >> fsp-observer-output-clean_${PROMETHEUS_HOST}_${PROMETHEUS_PORT}.txt
  echo -e "flare_fsp_address_balance_wei_signatures:$( grep "flare_fsp_address_balance_wei.*signatures" fsp-observer-output_${PROMETHEUS_HOST}_${PROMETHEUS_PORT}.txt | awk '{print $NF}' )" >> fsp-observer-output-clean_${PROMETHEUS_HOST}_${PROMETHEUS_PORT}.txt
  echo -e "flare_fsp_address_balance_wei_policy:$( grep "flare_fsp_address_balance_wei.*policy" fsp-observer-output_${PROMETHEUS_HOST}_${PROMETHEUS_PORT}.txt | awk '{print $NF}' )" >> fsp-observer-output-clean_${PROMETHEUS_HOST}_${PROMETHEUS_PORT}.txt
  echo -e "flare_fsp_address_balance_wei_fast_updates_1:$( grep "flare_fsp_address_balance_wei.*fast" fsp-observer-output_${PROMETHEUS_HOST}_${PROMETHEUS_PORT}.txt | sed -n '1p' | awk '{print $NF}' )" >> fsp-observer-output-clean_${PROMETHEUS_HOST}_${PROMETHEUS_PORT}.txt
  echo -e "flare_fsp_address_balance_wei_fast_updates_2:$( grep "flare_fsp_address_balance_wei.*fast" fsp-observer-output_${PROMETHEUS_HOST}_${PROMETHEUS_PORT}.txt | sed -n '2p' | awk '{print $NF}' )" >> fsp-observer-output-clean_${PROMETHEUS_HOST}_${PROMETHEUS_PORT}.txt
  echo -e "flare_fsp_address_balance_wei_fast_updates_3:$( grep "flare_fsp_address_balance_wei.*fast" fsp-observer-output_${PROMETHEUS_HOST}_${PROMETHEUS_PORT}.txt | sed -n '3p' | awk '{print $NF}' )" >> fsp-observer-output-clean_${PROMETHEUS_HOST}_${PROMETHEUS_PORT}.txt
  # flare_fsp_registered_current_epoch
  echo -e "flare_fsp_registered_current_epoch:$( grep "flare_fsp_registered_current_epoch" fsp-observer-output_${PROMETHEUS_HOST}_${PROMETHEUS_PORT}.txt | awk '{print $NF}' )" >> fsp-observer-output-clean_${PROMETHEUS_HOST}_${PROMETHEUS_PORT}.txt
  # flare_fsp_registered_next_epoch
  echo -e "flare_fsp_registered_next_epoch:$( grep "flare_fsp_registered_next_epoch" fsp-observer-output_${PROMETHEUS_HOST}_${PROMETHEUS_PORT}.txt | awk '{print $NF}' )" >> fsp-observer-output-clean_${PROMETHEUS_HOST}_${PROMETHEUS_PORT}.txt
  # flare_fsp_voting_round
  echo -e "flare_fsp_voting_round:$( grep "flare_fsp_voting_round" fsp-observer-output_${PROMETHEUS_HOST}_${PROMETHEUS_PORT}.txt | awk '{print $NF}' )" >> fsp-observer-output-clean_${PROMETHEUS_HOST}_${PROMETHEUS_PORT}.txt
  # flare_fsp_reward_epoch
  echo -e "flare_fsp_reward_epoch:$( grep "flare_fsp_reward_epoch" fsp-observer-output_${PROMETHEUS_HOST}_${PROMETHEUS_PORT}.txt | awk '{print $NF}' )" >> fsp-observer-output-clean_${PROMETHEUS_HOST}_${PROMETHEUS_PORT}.txt
  # flare_fsp_node_uptime_ratio
  echo -e "flare_fsp_node_uptime_ratio:$( grep "flare_fsp_node_uptime_ratio" fsp-observer-output_${PROMETHEUS_HOST}_${PROMETHEUS_PORT}.txt | awk '{print $NF}' )" >> fsp-observer-output-clean_${PROMETHEUS_HOST}_${PROMETHEUS_PORT}.txt
  # flare_fsp_fast_update_blocks_since_last
  echo -e "flare_fsp_fast_update_blocks_since_last:$( grep "flare_fsp_fast_update_blocks_since_last" fsp-observer-output_${PROMETHEUS_HOST}_${PROMETHEUS_PORT}.txt | awk '{print $NF}' )" >> fsp-observer-output-clean_${PROMETHEUS_HOST}_${PROMETHEUS_PORT}.txt
  # flare_fsp_ftso_anchor_feeds_success_rate_bips
  echo -e "flare_fsp_ftso_anchor_feeds_success_rate_bips:$( grep "flare_fsp_ftso_anchor_feeds_success_rate_bips" fsp-observer-output_${PROMETHEUS_HOST}_${PROMETHEUS_PORT}.txt | awk '{print $NF}' )" >> fsp-observer-output-clean_${PROMETHEUS_HOST}_${PROMETHEUS_PORT}.txt
  # flare_fsp_fdc_participation_rate_bips
  echo -e "flare_fsp_fdc_participation_rate_bips:$( grep "flare_fsp_fdc_participation_rate_bips" fsp-observer-output_${PROMETHEUS_HOST}_${PROMETHEUS_PORT}.txt | awk '{print $NF}' )" >> fsp-observer-output-clean_${PROMETHEUS_HOST}_${PROMETHEUS_PORT}.txt
  # flare_fsp_reveal_offence_total
  echo -e "flare_fsp_reveal_offence_total_ftso:$( grep "flare_fsp_reveal_offence_total.*ftso" fsp-observer-output_${PROMETHEUS_HOST}_${PROMETHEUS_PORT}.txt | awk '{print $NF}' )" >> fsp-observer-output-clean_${PROMETHEUS_HOST}_${PROMETHEUS_PORT}.txt
  echo -e "flare_fsp_reveal_offence_total_fdc:$( grep "flare_fsp_reveal_offence_total.*fdc" fsp-observer-output_${PROMETHEUS_HOST}_${PROMETHEUS_PORT}.txt | awk '{print $NF}' )" >> fsp-observer-output-clean_${PROMETHEUS_HOST}_${PROMETHEUS_PORT}.txt
  # flare_fsp_signature_grace_period_missed_total
  echo -e "flare_fsp_signature_grace_period_missed_total_ftso:$( grep "flare_fsp_signature_grace_period_missed_total.*ftso" fsp-observer-output_${PROMETHEUS_HOST}_${PROMETHEUS_PORT}.txt | awk '{print $NF}' )" >> fsp-observer-output-clean_${PROMETHEUS_HOST}_${PROMETHEUS_PORT}.txt
  echo -e "flare_fsp_signature_grace_period_missed_total_fdc:$( grep "flare_fsp_signature_grace_period_missed_total.*fdc" fsp-observer-output_${PROMETHEUS_HOST}_${PROMETHEUS_PORT}.txt | awk '{print $NF}' )" >> fsp-observer-output-clean_${PROMETHEUS_HOST}_${PROMETHEUS_PORT}.txt
  # flare_fsp_signature_mismatch_total
  echo -e "flare_fsp_signature_mismatch_total_ftso:$( grep "flare_fsp_signature_mismatch_total.*ftso" fsp-observer-output_${PROMETHEUS_HOST}_${PROMETHEUS_PORT}.txt | awk '{print $NF}' )" >> fsp-observer-output-clean_${PROMETHEUS_HOST}_${PROMETHEUS_PORT}.txt
  echo -e "flare_fsp_signature_mismatch_total_fdc:$( grep "flare_fsp_signature_mismatch_total.*fdc" fsp-observer-output_${PROMETHEUS_HOST}_${PROMETHEUS_PORT}.txt | awk '{print $NF}' )" >> fsp-observer-output-clean_${PROMETHEUS_HOST}_${PROMETHEUS_PORT}.txt
  # flare_fsp_contract_address_wrong_total
  echo -e "flare_fsp_contract_address_wrong_total_submission:$( grep "flare_fsp_contract_address_wrong_total.*submission" fsp-observer-output_${PROMETHEUS_HOST}_${PROMETHEUS_PORT}.txt | awk '{print $NF}' )" >> fsp-observer-output-clean_${PROMETHEUS_HOST}_${PROMETHEUS_PORT}.txt
  echo -e "flare_fsp_contract_address_wrong_total_relay:$( grep "flare_fsp_contract_address_wrong_total.*relay" fsp-observer-output_${PROMETHEUS_HOST}_${PROMETHEUS_PORT}.txt | awk '{print $NF}' )" >> fsp-observer-output-clean_${PROMETHEUS_HOST}_${PROMETHEUS_PORT}.txt
  # flare_fsp_unclaimed_rewards_wei
  #echo -e "flare_fsp_unclaimed_rewards_wei:$( grep "flare_fsp_unclaimed_rewards_wei" fsp-observer-output_${PROMETHEUS_HOST}_${PROMETHEUS_PORT}.txt | awk '{print $NF}' )" >> fsp-observer-output-clean_${PROMETHEUS_HOST}_${PROMETHEUS_PORT}.txt

  jq -Rn '[inputs | split(":") | {(.[0]): .[1]}] | add' fsp-observer-output-clean_${PROMETHEUS_HOST}_${PROMETHEUS_PORT}.txt > fsp-observer-output-clean_${PROMETHEUS_HOST}_${PROMETHEUS_PORT}.json
  
  rm -f fsp-observer-output_${PROMETHEUS_HOST}_${PROMETHEUS_PORT}.txt
  rm -f fsp-observer-output-clean_${PROMETHEUS_HOST}_${PROMETHEUS_PORT}.txt
  
  rm -f zbx_fsp_observer_json_query_${PROMETHEUS_HOST}_${PROMETHEUS_PORT}.lock
fi

# If you wish to avoid JSON preprocessing within the Zabbix Item, remove the comment on the next line, and comment out the following line.
#output=`cat fsp-observer-output-clean_${PROMETHEUS_HOST}_${PROMETHEUS_PORT}.json | jq .${1}`
output=`cat fsp-observer-output-clean_${PROMETHEUS_HOST}_${PROMETHEUS_PORT}.json`

echo ${output}
