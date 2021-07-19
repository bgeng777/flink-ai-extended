rm -r ../docs/source_rst/
mkdir ../docs/source_rst/
mkdir __tmp
sphinx-apidoc -f -e -M -o ../docs/__tmp/ ../ai_flow ../ai_flow/*test*
cp -a __tmp/. ../docs/source_rst/
rm __tmp/*
sphinx-apidoc -f -e -M -o ../docs/__tmp/ ../ai_flow_plugins ../ai_flow_plugins/*test*
cp -a __tmp/. ../docs/source_rst/
rm -r __tmp
make clean html