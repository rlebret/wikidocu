package = "wikidocu"
version = "scm-1"

source = {
   url = "git://github.com/rlebret/wikidocu",
   branch = "master",
}

description = {
   summary = "Wikipedia parsed articles management",
   detailed = [[
   ]],
   homepage= "https://github.com/rlebret/wikidocu",
   license = "BSD"
}

dependencies = {
   "torch >= 7.0",
   "utils >= 1.0"
}

build = {
   type = "builtin",
   modules = {
      ["wikidocu.init"] = "init.lua"
   }
}
