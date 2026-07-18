# FSP-OBSERVER - ZABBIX INTEGRATION

## Requirements

To integrate Zabbix with the fsp-observer, you will need to install [`jq`](https://jqlang.org/) on the system where the Zabbix Agent is running.

## Installation

To install the Zabbix Integration tools for the fsp-observer, please do the following:

- Copy the contents of the `bin/` directory to the `/usr/local/bin/` directory of the system where the Zabbix Agent is running,
- Copy the `conf/zabbix_agentd.userparams.conf` file to the `conf.d` directory of your Zabbix installation, or include it in the main Zabbix configuration file,
- Restart the Zabbix Agent,
- Import the zbx_export_fsp_observer_template.yaml template file via the Web interface of your Zabbix installation.

You should now be able to add the fsp-observer host, with its {$HOST.NAME} and {$HOST.PORT} macros. Each host having its own macros, you will be able
to connect Zabbix to multiple instances of the Prometheus server (fsp-observer), if needed.

## Troubleshooting

You can always check if the Zabbix Agent has recognized the custom Zabbix Item, and has the necessary permissions to run it, in order to connect to the
Prometheus server (localhost:8000 in the following example), by running these commands on the CLI:

```
zabbix_agentd -p | grep fspobs
zabbix_agentd -t fspobs.get["flare_fsp_submit_ok_total_submit1",localhost,8000]
```
