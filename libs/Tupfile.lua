fmt_includes = {
    '-Ifmt/include',
}
fmt_sources = {
    'fmt/src/format.cc',
    'fmt/src/os.cc',
}
fmt_lib = tup.foreach_rule(fmt_sources, CC_PATH..'g++ $(fmt_includes) -c %f -o %o', {'build/obj/%B.o', extra_outputs = ROOT..'/<libfmt>'})
