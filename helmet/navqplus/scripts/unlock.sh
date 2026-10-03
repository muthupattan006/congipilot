if [ -z "$TMUX" ] ; then
	keychain -q ~/.ssh/id_rsa;
fi
. ~/.keychain/$(hostname)-sh 2> /dev/null

